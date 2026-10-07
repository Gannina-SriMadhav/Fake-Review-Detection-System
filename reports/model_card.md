# Model Card — Linear SVM + TF-IDF

> This system provides an automated prediction and should not be treated as definitive proof that a review is fraudulent.

| Field | Value |
|---|---|
| Model version | 2026.10.07.0655 |
| Training date (UTC) | 2026-10-07T06:55:11.593584Z |
| Algorithm | Linear SVM + TF-IDF (probabilities calibrated: sigmoid) |
| Training dataset | `data/fake reviews dataset.csv` (40,391 rows after cleaning; {'GENUINE': 20215, 'FAKE': 20176}) |
| Split | train 28,850 / validation 5,770 / test 5,771 (stratified, near-duplicate-group aware) |
| Model selection | best f1_fake on the **validation** split |
| Features | word 1-2-gram TF-IDF, char 2-5-gram TF-IDF, 18 stylistic/sentiment features (model-learned weights) |

## Held-out test metrics (positive class = FAKE)
Accuracy 0.9574 · Precision 0.9486 · Recall 0.9670 · F1 0.9577 ·
ROC-AUC 0.9910 · PR-AUC 0.9909 · Brier 0.0345 · ECE 0.0165

Confusion (test): TN 2737 · FP 151 (genuine flagged FAKE) · FN 95 (fake passed as GENUINE) · TP 2788

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
- Test-set error rates by length bucket: {'<=5': {'n': 2, 'error_rate': 0.5}, '6-15': {'n': 557, 'error_rate': 0.1634}, '16-40': {'n': 2394, 'error_rate': 0.0547}, '41-100': {'n': 1595, 'error_rate': 0.0132}, '>100': {'n': 1223, 'error_rate': 0.0016}}

## Potential biases
Writing style, dialect, non-native English and informal grammar can correlate with the label in the training data; the model
may mis-flag such genuine reviews. Check false-positive examples in `reports/evaluation.json` before relying on predictions.

## Failure cases
151 false positives and 95 false negatives on the test split; see `error_analysis` in `reports/evaluation.json`.
