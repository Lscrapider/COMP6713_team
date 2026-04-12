import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput


LABELS = [
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "donation_and_volunteering",
    "infrastructure_and_utilities_damage",
    "injured_or_dead_people",
    "missing_trapped_or_found_people",
    "not_related_or_irrelevant",
    "other_useful_information",
    "sympathy_and_emotional_support",
]


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


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def build_prediction_frame(predictions: np.ndarray, frame: pd.DataFrame, id_to_label: Dict[int, str]) -> pd.DataFrame:
    predicted_ids = np.argmax(predictions, axis=-1)
    return pd.DataFrame(
        {
            "id": np.arange(1, len(frame) + 1),
            "true_label": frame["label"].tolist(),
            "prediction": [id_to_label[idx] for idx in predicted_ids],
        }
    )


def save_metrics(output_dir: Path, split_name: str, true_labels: List[str], pred_labels: List[str]) -> None:
    label_report = classification_report(
        true_labels,
        pred_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, pred_labels, labels=LABELS)
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "split": split_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "labels": LABELS,
        "classification_report": label_report,
        "confusion_matrix": matrix.tolist(),
    }
    with (output_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def write_predictions_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "true_label", "prediction"])
        writer.writerows(frame.itertuples(index=False, name=None))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune bert-base-uncased on the disaster tweet dataset with CrisisLex features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert_finetuned"))
    parser.add_argument("--lexicon-path", type=Path, default=Path("data/CrisisLexLexicon/CrisisLexRec.txt"))
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_datasets(args.data_dir)
    lexicon = load_lexicon(args.lexicon_path)
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    feature_dim = len(extract_lexicon_features("", lexicon))
    model = BertWithLexiconFeatures(
        model_name=args.model_name,
        num_labels=len(LABELS),
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        feature_dim=feature_dim,
    )

    train_dataset = TweetDataset(data.train, tokenizer, label_to_id, args.max_length, lexicon)
    val_dataset = TweetDataset(data.val, tokenizer, label_to_id, args.max_length, lexicon)
    test_dataset = TweetDataset(data.test, tokenizer, label_to_id, args.max_length, lexicon)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=LexiconFeatureCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "best_model"))
    tokenizer.save_pretrained(str(args.output_dir / "best_model"))

    val_predictions = trainer.predict(val_dataset)
    val_frame = build_prediction_frame(val_predictions.predictions, data.val, id_to_label)
    save_metrics(args.output_dir, "val", val_frame["true_label"].tolist(), val_frame["prediction"].tolist())
    write_predictions_csv(args.output_dir / "val_predictions.csv", val_frame)

    test_predictions = trainer.predict(test_dataset)
    test_frame = build_prediction_frame(test_predictions.predictions, data.test, id_to_label)
    save_metrics(args.output_dir, "test", test_frame["true_label"].tolist(), test_frame["prediction"].tolist())
    write_predictions_csv(args.output_dir / "bert_finetuned_predictions.csv", test_frame)


if __name__ == "__main__":
    main()
