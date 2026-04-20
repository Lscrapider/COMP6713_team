# Disaster Tweet Classification

Training scripts for the 9-class disaster tweet classification task using `bert-base-uncased` with CrisisLex lexicon features.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python train_loss_comparison.py --run-mode cross_entropy
```

Optional example with custom settings:

```bash
.venv/bin/python train_loss_comparison.py \
  --run-mode cross_entropy \
  --num-train-epochs 3 \
  --train-batch-size 16 \
  --eval-batch-size 32 \
  --max-length 128 \
  --fp16
```

Shrink all three splits to the same fraction for quick tests:

```bash
.venv/bin/python train_loss_comparison.py --run-mode cross_entropy --data-fraction 0.1
```

Run all loss-function experiments in one script:

```bash
.venv/bin/python train_loss_comparison.py --run-mode all
```

This writes four separate experiment folders:

- `outputs/bert_cross_entropy/`
- `outputs/bert_weighted_cross_entropy/`
- `outputs/bert_label_smoothing/`
- `outputs/bert_weighted_label_smoothing/`

## Outputs

Cross-entropy baseline files are written to `outputs/bert_cross_entropy/`:

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
