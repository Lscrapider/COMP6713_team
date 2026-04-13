import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from model_config import LABELS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class CrisisLexicon:
    terms: Sequence[str]
    unigram_terms: set[str]
    phrase_terms: Sequence[str]


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_columns = {"text", "label"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str)
    unknown_labels = sorted(set(df["label"]) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"{path} contains unexpected labels: {unknown_labels}")
    return df


def load_datasets(data_dir: Path) -> DatasetBundle:
    return DatasetBundle(
        train=load_split(data_dir / "train.csv"),
        val=load_split(data_dir / "val.csv"),
        test=load_split(data_dir / "test.csv"),
    )


def sample_split(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return frame
    sample_size = max(1, int(len(frame) * fraction))
    return frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def sample_datasets(bundle: DatasetBundle, fraction: float, seed: int) -> DatasetBundle:
    return DatasetBundle(
        train=sample_split(bundle.train, fraction, seed),
        val=sample_split(bundle.val, fraction, seed),
        test=sample_split(bundle.test, fraction, seed),
    )


def load_lexicon(path: Path) -> CrisisLexicon:
    with path.open(encoding="utf-8") as f:
        terms = [line.strip().lower() for line in f if line.strip()]
    unigram_terms = {term for term in terms if " " not in term}
    phrase_terms = [term for term in terms if " " in term]
    return CrisisLexicon(terms=terms, unigram_terms=unigram_terms, phrase_terms=phrase_terms)


def extract_lexicon_features(text: str, lexicon: CrisisLexicon) -> List[float]:
    lowered = text.lower()
    tokens = re.findall(r"\b\w+\b", lowered)
    token_count = max(len(tokens), 1)

    matched_unigrams = {token for token in tokens if token in lexicon.unigram_terms}
    matched_phrases = [term for term in lexicon.phrase_terms if term in lowered]
    unique_terms = matched_unigrams | set(matched_phrases)

    return [
        float(len(matched_unigrams)),
        float(len(matched_phrases)),
        float(len(unique_terms)),
        float(len(unique_terms) / token_count),
        float(bool(unique_terms)),
    ]


class TweetDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        label_to_id: Dict[str, int],
        max_length: int,
        lexicon: CrisisLexicon,
    ):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.lexicon = lexicon

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
        )
        encoding["labels"] = self.label_to_id[row["label"]]
        encoding["lexicon_features"] = extract_lexicon_features(row["text"], self.lexicon)
        return encoding


class LexiconFeatureCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        text_features = [
            {k: v for k, v in feature.items() if k not in {"labels", "lexicon_features"}}
            for feature in features
        ]
        batch = self.tokenizer.pad(text_features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        batch["lexicon_features"] = torch.tensor(
            [feature["lexicon_features"] for feature in features],
            dtype=torch.float,
        )
        return batch


class BertWithLexiconFeatures(nn.Module):
    def __init__(self, model_name: str, num_labels: int, label_to_id: Dict[str, int], id_to_label: Dict[int, str], feature_dim: int):
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id_to_label,
            label2id=label_to_id,
        )
        self.config = config
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + feature_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, lexicon_features=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        combined = torch.cat([pooled_output, lexicon_features], dim=-1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
