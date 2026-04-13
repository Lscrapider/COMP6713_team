# Disaster Tweet Classification

Baseline training script for the 9-class disaster tweet classification task using `bert-base-uncased` with CrisisLex lexicon features.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 train_bert_baseline.py
```

Optional example with custom settings:

```bash
.venv/bin/python train_bert_baseline.py \
  --num-train-epochs 3 \
  --train-batch-size 16 \
  --eval-batch-size 32 \
  --max-length 128 \
  --fp16
```

Shrink all three splits to the same fraction for quick tests:

```bash
.venv/bin/python train_bert_baseline.py --data-fraction 0.1
```

## Outputs

Files are written to `outputs/bert_finetuned/`:

- `best_model/`: saved tokenizer and best checkpoint
- `val_metrics.json`: validation metrics and confusion matrix
- `test_metrics.json`: test metrics and confusion matrix
- `val_predictions.csv`: validation predictions
- `bert_finetuned_predictions.csv`: test predictions in `id,true_label,prediction` format

## Predict With The Trained Model

Classify a single text with the saved model:

```bash
.venv/bin/python predict_bert_finetuned.py --text "earthquake in city center"
```

Run the saved model on the default test set and also compute metrics:

```bash
.venv/bin/python predict_bert_finetuned.py
```

This writes:

- `outputs/bert_finetuned/test_predictions.csv`
- `outputs/bert_finetuned/test_inference_metrics.json`

Batch predict a custom CSV containing a `text` column:

```bash
.venv/bin/python predict_bert_finetuned.py \
  --input-csv data/test.csv \
  --output-csv outputs/test_predictions.csv \
  --metrics-output outputs/test_metrics.json
```

## Data Assumptions

The script expects:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

Each file must contain:

- `text`
- `label`

The script also uses:

- `data/CrisisLexLexicon/CrisisLexRec.txt`

For each tweet, it extracts a small set of CrisisLex-based features and concatenates them with the transformer representation before classification.
