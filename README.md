# Laptop Analytics Dashboard with Classification

A Streamlit application for descriptive laptop analytics and supervised classification.

## Classification algorithms

The Classification Models tab compares four algorithms:

1. **Gradient Boosting Classifier** — builds an additive ensemble of decision trees.
2. **Random Forest Classifier** — combines many randomized decision trees.
3. **Support Vector Machine (SVM)** — uses an RBF kernel to separate classes.
4. **Gaussian Naive Bayes** — applies Bayes' rule with a Gaussian likelihood assumption for numeric transformed features.

## Classification target

The default target is **Price Segment**, derived from the numeric `Price` field using quantile-based grouping:

- Budget
- Mid-range
- Premium

If there are only two usable quantile groups, the app uses Budget and Premium. The original `Price` field is excluded from predictors for this derived target to prevent target leakage.

You can alternatively choose an existing categorical column with 2–15 distinct values as the target.

## Evaluation

All four models use the same stratified train/test split and report:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Macro F1-score
- Confusion matrix
- Per-class precision, recall, and F1-score
- Test-set predictions

Weighted F1 accounts for class frequency. Macro F1 gives each class equal weight and is useful when class balance matters. Higher scores are generally better, but the results depend on the dataset, selected target, features, and split.

## Project structure

```text
laptop_analytics_classification/
├── app.py
├── ml_classification.py
├── requirements.txt
└── README.md
```

## Run on macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown in the terminal, normally `http://localhost:8501`.

## Dataset

By default, the application loads:

`https://raw.githubusercontent.com/drpatrickcerna-prog/laptop/main/laptop.csv`

You can upload another compatible CSV from the sidebar.

Expected default fields include `Brand`, `Name`, `Price`, processor, RAM, display, GPU, and storage specifications. The app handles optional columns where possible.

## Important methodological notes

- Classification requires a categorical target. Price is numeric, so the default workflow derives price segments using quantiles.
- For the derived Price Segment target, `Price` is removed from the predictors to avoid target leakage.
- Rows with missing target labels are excluded. Predictor missing values are imputed within each training pipeline.
- One-hot encoded categorical data is converted to dense output because Gaussian Naive Bayes requires dense input. This is appropriate for a modest laptop dataset; for very large/high-cardinality data, consider reducing categories or using a sparse-compatible Naive Bayes implementation.
- At least two records per class are required for stratified train/test evaluation.
- The reported test scores are not a substitute for cross-validation, external validation, or careful review of class imbalance.
- Price quantile groups are relative to the dataset, not universal market price bands.

## Previous regression module

This version focuses the machine-learning tab on classification. The descriptive analytics, filters, scatterplot explorer, data-quality profile, and CSV downloads remain available.
