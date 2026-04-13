import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from disaster_tweet_model import (
    BertWithLexiconFeatures,
    LexiconFeatureCollator,
    TweetDataset,
    extract_lexicon_features,
    load_datasets,
    load_lexicon,
    sample_datasets,
    set_seed,
)
from model_config import (
    DEFAULT_DATA_DIR,
    DEFAULT_LEXICON_PATH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    LABELS,
    build_label_mappings,
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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--data-fraction", type=float, default=1.0, help="Fraction of each split to keep, in the range (0, 1].")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.data_fraction <= 1.0:
        raise ValueError("--data-fraction must be in the range (0, 1].")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_datasets(args.data_dir)
    data = sample_datasets(data, args.data_fraction, args.seed)
    lexicon = load_lexicon(args.lexicon_path)
    label_to_id, id_to_label = build_label_mappings()

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
        eval_strategy="epoch",
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
        processing_class=tokenizer,
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
