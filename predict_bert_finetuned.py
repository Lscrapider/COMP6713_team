import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer

from disaster_tweet_model import BertWithLexiconFeatures, extract_lexicon_features, load_lexicon
from model_config import (
    DEFAULT_BEST_MODEL_DIR,
    DEFAULT_LEXICON_PATH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEST_INPUT_CSV,
    DEFAULT_TEST_METRICS_JSON,
    DEFAULT_TEST_PREDICTIONS_CSV,
    LABELS,
    build_label_mappings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned disaster tweet model.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_BEST_MODEL_DIR)
    parser.add_argument("--base-model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--text", type=str, help="Single text to classify.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_TEST_INPUT_CSV, help="CSV file containing a text column to classify.")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_TEST_PREDICTIONS_CSV, help="Where to save batch predictions.")
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_TEST_METRICS_JSON, help="Where to save evaluation metrics when labels are available.")
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_weights(model, model_dir: Path) -> None:
    safetensors_path = model_dir / "model.safetensors"
    pytorch_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path))
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {model_dir}")

    model.load_state_dict(state_dict)


def load_inference_bundle(args: argparse.Namespace):
    label_to_id, id_to_label = build_label_mappings()
    lexicon = load_lexicon(args.lexicon_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    feature_dim = len(extract_lexicon_features("", lexicon))
    model = BertWithLexiconFeatures(
        model_name=args.base_model_name,
        num_labels=len(LABELS),
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        feature_dim=feature_dim,
    )
    load_model_weights(model, args.model_dir)
    device = pick_device()
    model.to(device)
    model.eval()
    return model, tokenizer, lexicon, id_to_label, device


def predict_texts(texts, model, tokenizer, lexicon, id_to_label, device, max_length: int):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    lexicon_features = torch.tensor(
        [extract_lexicon_features(text, lexicon) for text in texts],
        dtype=torch.float,
    )
    encodings = {key: value.to(device) for key, value in encodings.items()}
    lexicon_features = lexicon_features.to(device)

    with torch.no_grad():
        outputs = model(**encodings, lexicon_features=lexicon_features)
        predicted_ids = outputs.logits.argmax(dim=-1).cpu().tolist()
    return [id_to_label[idx] for idx in predicted_ids]


def run_single_text(args: argparse.Namespace, model, tokenizer, lexicon, id_to_label, device) -> None:
    prediction = predict_texts(
        [args.text],
        model,
        tokenizer,
        lexicon,
        id_to_label,
        device,
        args.max_length,
    )[0]
    print(prediction)


def build_metrics(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "labels": LABELS,
        "classification_report": classification_report(
            true_labels,
            pred_labels,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(true_labels, pred_labels, labels=LABELS).tolist(),
    }


def run_batch(args: argparse.Namespace, model, tokenizer, lexicon, id_to_label, device) -> None:
    frame = pd.read_csv(args.input_csv)
    if args.text_column not in frame.columns:
        raise ValueError(f"{args.input_csv} is missing column: {args.text_column}")

    texts = frame[args.text_column].fillna("").astype(str).tolist()
    predictions = predict_texts(
        texts,
        model,
        tokenizer,
        lexicon,
        id_to_label,
        device,
        args.max_length,
    )
    output = frame.copy()
    output["prediction"] = predictions

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

    if args.label_column in frame.columns:
        true_labels = frame[args.label_column].fillna("").astype(str).tolist()
        metrics = build_metrics(true_labels, predictions)
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.metrics_output}")
        print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}, indent=2))


def main() -> None:
    args = parse_args()
    model, tokenizer, lexicon, id_to_label, device = load_inference_bundle(args)

    if args.text:
        run_single_text(args, model, tokenizer, lexicon, id_to_label, device)
    else:
        run_batch(args, model, tokenizer, lexicon, id_to_label, device)


if __name__ == "__main__":
    main()
