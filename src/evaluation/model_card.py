"""Model card generated from real run outputs."""
from __future__ import annotations

from pathlib import Path


def write_model_card(meta: dict, res: dict, data: dict, ea: dict, path: Path) -> None:
    t = res["test"]
    c = t["confusion"]
    md = f"""# Model Card — {meta['model_name']}

> This system provides an automated prediction and should not be treated as definitive proof that a review is fraudulent.

| Field | Value |
|---|---|
| Model version | {meta['version']} |
| Training date (UTC) | {meta['trained_at']} |
| Algorithm | {meta['model_name']} (probabilities calibrated: {meta['calibration']}) |
| Training dataset | `{meta['dataset']}` ({data['clean_rows']:,} rows after cleaning; {data['label_counts']}) |
| Split | train {meta['n_train']:,} / validation {meta['n_val']:,} / test {meta['n_test']:,} (stratified, near-duplicate-group aware) |
| Model selection | best {meta['selection_metric']} on the **validation** split |
| Features | word 1-2-gram TF-IDF, char 2-5-gram TF-IDF, {len(meta['style_features'])} stylistic/sentiment features (model-learned weights) |

## Held-out test metrics (positive class = FAKE)
Accuracy {t['accuracy']:.4f} · Precision {t['precision_fake']:.4f} · Recall {t['recall_fake']:.4f} · F1 {t['f1_fake']:.4f} ·
ROC-AUC {t['roc_auc']:.4f} · PR-AUC {t['pr_auc']:.4f} · Brier {t['brier']:.4f} · ECE {t['ece']:.4f}

Confusion (test): TN {c['tn']} · FP {c['fp']} (genuine flagged FAKE) · FN {c['fn']} (fake passed as GENUINE) · TP {c['tp']}

## Intended use
Triage / decision support: surface reviews that look statistically similar to the labelled "fake" class so a human can inspect them.

## Not intended for
Automatic removal of reviews, penalising sellers/users, legal or reputational claims, or any decision without human review.

## Known limitations
- The labels in this dataset are *computer-generated (CG)* vs *original (OR)* reviews. The model therefore learns to separate
  machine-generated text from human text in these 10 product categories; it is **not** validated on paid/incentivised human fake
  reviews, other languages, or other domains. Scores on other data may be much lower.
- Very short reviews carry little evidence; the model should (and the UI does) report lower confidence for them.
- Calibration was fit on this dataset's class balance (~50/50); real-world prevalence of fakes is different, so absolute
  probabilities will not transfer without recalibration.
- Test-set error rates by length bucket: {ea['error_rate_by_length']}

## Potential biases
Writing style, dialect, non-native English and informal grammar can correlate with the label in the training data; the model
may mis-flag such genuine reviews. Check false-positive examples in `reports/evaluation.json` before relying on predictions.

## Failure cases
{ea['n_false_positive']} false positives and {ea['n_false_negative']} false negatives on the test split; see `error_analysis` in `reports/evaluation.json`.
"""
    path.write_text(md, encoding="utf-8")
