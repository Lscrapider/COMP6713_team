import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

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
    DEFAULT_BASELINE_OUTPUT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_EXTENSION_OUTPUT_DIR,
    DEFAULT_LABEL_SMOOTHING_OUTPUT_DIR,
    DEFAULT_LEXICON_PATH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_WEIGHTED_OUTPUT_DIR,
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
    # Save both summary metrics and per-class details for later report tables.
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


def compute_class_weights(train_frame: pd.DataFrame, label_to_id: Dict[str, int]) -> torch.Tensor:
    counts = train_frame["label"].value_counts()
    # Inverse-frequency weighting gives rare classes a larger loss contribution.
    weights = torch.tensor(
        [1.0 / counts[label] for label in label_to_id],
        dtype=torch.float,
    )
    return weights / weights.sum() * len(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERT loss-function experiments for disaster tweet classification.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing train.csv, val.csv, and test.csv.")
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH, help="CrisisLex lexicon file used to extract extra features.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face base model name or local model path.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum token length for each tweet.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Per-device batch size used during training.")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Per-device batch size used during validation and test prediction.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay applied by the optimizer.")
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Number of full passes over the training set.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Fraction of training steps used for learning-rate warmup.")
    parser.add_argument("--data-fraction", type=float, default=1.0, help="Fraction of each split to use; set below 1.0 for quick tests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling and training setup.")
    parser.add_argument("--logging-steps", type=int, default=100, help="Number of training steps between log messages.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision when supported.")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision when supported.")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing strength for label-smoothing loss modes.")
    parser.add_argument(
        "--run-mode",
        type=str,
        default="weighted_cross_entropy",
        choices=["all", "cross_entropy", "weighted_cross_entropy", "label_smoothing", "weighted_label_smoothing"],
        help="Experiment to run.",
    )
    parser.add_argument("--baseline-output-dir", type=Path, default=DEFAULT_BASELINE_OUTPUT_DIR, help="Output directory for cross-entropy baseline.")
    parser.add_argument("--weighted-output-dir", type=Path, default=DEFAULT_WEIGHTED_OUTPUT_DIR, help="Output directory for weighted cross entropy.")
    parser.add_argument("--label-smoothing-output-dir", type=Path, default=DEFAULT_LABEL_SMOOTHING_OUTPUT_DIR, help="Output directory for label smoothing.")
    parser.add_argument("--extension-output-dir", type=Path, default=DEFAULT_EXTENSION_OUTPUT_DIR, help="Output directory for weighted label smoothing.")
    return parser.parse_args()


def build_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    use_mps = torch.backends.mps.is_available()
    # Pin memory is useful for CUDA but can cause issues or no benefit on Apple MPS.
    return TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
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
        dataloader_num_workers=2,
        dataloader_pin_memory=not use_mps,
        report_to="none",
    )


class ExperimentLoggingCallback(TrainerCallback):
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

    def on_train_begin(self, args, state, control, **kwargs):
        print(
            f"[train] {self.experiment_name} | epochs={args.num_train_epochs:g} "
            f"| max_steps={state.max_steps} | output_dir={args.output_dir}"
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"[epoch] {self.experiment_name} | starting epoch {state.epoch or 0:.2f}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        epoch = logs.get("epoch", state.epoch)
        step = state.global_step
        max_steps = state.max_steps
        print(
            f"[log] {self.experiment_name} | epoch {epoch:.2f} | "
            f"step {step}/{max_steps} | loss={logs['loss']:.4f}"
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"[epoch] {self.experiment_name} | finished epoch {state.epoch or 0:.2f}")


def run_experiment(
    experiment_name: str,
    output_dir: Path,
    loss_type: str,
    args: argparse.Namespace,
    data,
    lexicon,
    tokenizer,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dim = len(extract_lexicon_features("", lexicon))
    class_weights = None
    if loss_type in {"weighted_cross_entropy", "weighted_label_smoothing"}:
        # Weighted loss modes compute class weights from the current training split.
        class_weights = compute_class_weights(data.train, label_to_id)

    model = BertWithLexiconFeatures(
        model_name=args.model_name,
        num_labels=len(LABELS),
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        feature_dim=feature_dim,
        loss_type=loss_type,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )

    train_dataset = TweetDataset(data.train, tokenizer, label_to_id, args.max_length, lexicon)
    val_dataset = TweetDataset(data.val, tokenizer, label_to_id, args.max_length, lexicon)
    test_dataset = TweetDataset(data.test, tokenizer, label_to_id, args.max_length, lexicon)

    trainer = Trainer(
        model=model,
        args=build_training_args(args, output_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=LexiconFeatureCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[ExperimentLoggingCallback(experiment_name)],
    )

    print(f"[experiment] starting {experiment_name} | loss_type={loss_type}")
    trainer.train()
    print(f"[experiment] finished training {experiment_name}; running validation/test predictions")
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        # Store the experiment setup next to the metrics and predictions.
        json.dump(
            {
                "experiment_name": experiment_name,
                "loss_type": loss_type,
                "label_smoothing": args.label_smoothing if "label_smoothing" in loss_type else 0.0,
                "uses_class_weights": loss_type in {"weighted_cross_entropy", "weighted_label_smoothing"},
            },
            f,
            indent=2,
        )

    val_predictions = trainer.predict(val_dataset)
    val_frame = build_prediction_frame(val_predictions.predictions, data.val, id_to_label)
    save_metrics(output_dir, "val", val_frame["true_label"].tolist(), val_frame["prediction"].tolist())
    write_predictions_csv(output_dir / "val_predictions.csv", val_frame)

    print(f"[experiment] {experiment_name} | validation complete; running test prediction")
    test_predictions = trainer.predict(test_dataset)
    test_frame = build_prediction_frame(test_predictions.predictions, data.test, id_to_label)
    save_metrics(output_dir, "test", test_frame["true_label"].tolist(), test_frame["prediction"].tolist())
    write_predictions_csv(output_dir / "test_predictions.csv", test_frame)
    print(f"[experiment] {experiment_name} | all outputs saved to {output_dir}")


def main() -> None:
    args = parse_args()
    if not 0 < args.data_fraction <= 1.0:
        raise ValueError("--data-fraction must be in the range (0, 1].")

    set_seed(args.seed)

    data = load_datasets(args.data_dir)
    data = sample_datasets(data, args.data_fraction, args.seed)
    lexicon = load_lexicon(args.lexicon_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    label_to_id, id_to_label = build_label_mappings()

    experiments = [
        ("baseline_cross_entropy", args.baseline_output_dir, "cross_entropy"),
        ("weighted_cross_entropy", args.weighted_output_dir, "weighted_cross_entropy"),
        ("label_smoothing", args.label_smoothing_output_dir, "label_smoothing"),
        ("extension_weighted_label_smoothing", args.extension_output_dir, "weighted_label_smoothing"),
    ]

    for experiment_name, output_dir, loss_type in experiments:
        # Skip experiments that are not selected by --run-mode.
        if args.run_mode not in {"all", loss_type}:
            continue
        run_experiment(
            experiment_name=experiment_name,
            output_dir=output_dir,
            loss_type=loss_type,
            args=args,
            data=data,
            lexicon=lexicon,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
        )


if __name__ == "__main__":
    main()
