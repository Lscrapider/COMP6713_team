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
python3 train_bert_baseline.py \
  --num-train-epochs 3 \
  --train-batch-size 16 \
  --eval-batch-size 32 \
  --max-length 128 \
  --fp16
```

## Outputs

Files are written to `outputs/bert_baseline/`:

- `best_model/`: saved tokenizer and best checkpoint
- `val_metrics.json`: validation metrics and confusion matrix
- `test_metrics.json`: test metrics and confusion matrix
- `val_predictions.csv`: validation predictions
- `bert_finetuned_predictions.csv`: test predictions in `id,true_label,prediction` format

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
