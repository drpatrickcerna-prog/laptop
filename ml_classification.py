"""Reusable classification functions for the Laptop Analytics Dashboard."""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def make_price_segment_target(
    data: pd.DataFrame,
    price_column: str = "Price",
) -> tuple[pd.DataFrame, str | None, str]:
    """Create a quantile-based price segment target without using Price as a feature."""
    frame = data.copy()
    if price_column not in frame.columns:
        return frame, None, f"Required target source column '{price_column}' was not found."

    numeric_price = pd.to_numeric(frame[price_column], errors="coerce")
    valid = numeric_price.dropna()
    if len(valid) < 6 or valid.nunique() < 2:
        return frame, None, "Not enough distinct, valid price values to derive price segments."

    # qcut avoids hard-coded currency thresholds and handles uneven price distributions.
    try:
        codes = pd.qcut(numeric_price, q=3, labels=False, duplicates="drop")
    except (ValueError, TypeError) as exc:
        return frame, None, f"Could not derive price segments: {exc}"

    number_of_bins = int(codes.max() + 1) if codes.notna().any() else 0
    if number_of_bins < 2:
        return frame, None, "Price values form fewer than two usable quantile groups."

    names_by_count = {
        2: ["Budget", "Premium"],
        3: ["Budget", "Mid-range", "Premium"],
    }
    labels = names_by_count.get(number_of_bins, [f"Segment {i + 1}" for i in range(number_of_bins)])
    frame["Price_Segment"] = codes.map(
        lambda value: labels[int(value)] if pd.notna(value) else np.nan
    )
    return frame, "Price_Segment", (
        f"Price Segment was derived from Price using {number_of_bins} quantile groups. "
        "The original Price column is excluded from predictors to prevent target leakage."
    )


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build dense, scaled preprocessing for numeric and categorical predictors.

    Dense one-hot output is intentional because GaussianNB requires dense input.
    For a very high-cardinality dataset, use a sparse-compatible alternative or
    reduce categorical cardinality before training.
    """
    categorical = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]

    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))

    if categorical:
        # sparse_output=False is supported by scikit-learn >= 1.2.
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    if not transformers:
        raise ValueError("No usable numeric or categorical predictors were selected.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_classification_models(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    test_fraction: float = 0.20,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Train four classifiers on one stratified hold-out split and return metrics/details."""
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is not present.")
    if not features:
        raise ValueError("Select at least one predictor.")
    if target in features:
        raise ValueError("The target column cannot also be a predictor.")
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Unknown predictor columns: {missing_features}")
    if not 0.10 <= float(test_fraction) <= 0.40:
        raise ValueError("Test fraction must be between 0.10 and 0.40.")

    frame = data[features + [target]].copy()
    frame[target] = frame[target].astype("string").str.strip()
    frame[target] = frame[target].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    frame = frame.dropna(subset=[target])
    frame[target] = frame[target].astype(str)

    # Avoid rows where every predictor is missing.
    frame = frame.loc[frame[features].notna().any(axis=1)].copy()
    if len(frame) < 10:
        raise ValueError("Not enough labeled records after removing rows without predictors.")

    class_counts = frame[target].value_counts()
    if class_counts.size < 2:
        raise ValueError("The target must contain at least two classes.")
    if int(class_counts.min()) < 2:
        rare = class_counts[class_counts < 2].to_dict()
        raise ValueError(f"Every class needs at least two records. Rare classes: {rare}")

    X = frame[features]
    y = frame[target]

    # Ensure both partitions can contain every class.
    n_classes = int(y.nunique())
    n_test = max(n_classes, int(ceil(len(frame) * float(test_fraction))))
    if n_test >= len(frame) - n_classes + 1:
        n_test = max(n_classes, int(round(len(frame) * float(test_fraction))))
    if n_test >= len(frame) or len(frame) - n_test < n_classes:
        raise ValueError(
            "The dataset is too small for a stratified train/test split with all classes represented."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=n_test,
        random_state=int(random_seed),
        stratify=y,
    )

    model_specs = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=int(random_seed),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=int(random_seed),
            n_jobs=-1,
        ),
        "Support Vector Machine (SVM)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=int(random_seed),
        ),
        "Naive Bayes (Gaussian)": GaussianNB(),
    }

    rows: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}
    for model_name, estimator in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(X_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)

        labels = sorted(y.unique().tolist())
        report = classification_report(
            y_test,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, predicted, labels=labels)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision (weighted)": precision_score(
                    y_test, predicted, average="weighted", zero_division=0
                ),
                "Recall (weighted)": recall_score(
                    y_test, predicted, average="weighted", zero_division=0
                ),
                "Weighted F1": f1_score(
                    y_test, predicted, average="weighted", zero_division=0
                ),
                "Macro F1": f1_score(
                    y_test, predicted, average="macro", zero_division=0
                ),
            }
        )

        prediction_details = pd.DataFrame(
            {
                "Actual class": y_test.to_numpy(),
                "Predicted class": predicted,
                "Correct": y_test.to_numpy() == predicted,
            },
            index=y_test.index,
        ).reset_index(names="Source row index")

        outputs[model_name] = {
            "pipeline": pipeline,
            "labels": labels,
            "confusion_matrix": cm,
            "classification_report": report,
            "prediction_details": prediction_details,
        }

    results = pd.DataFrame(rows).sort_values(
        ["Weighted F1", "Accuracy"], ascending=False
    ).reset_index(drop=True)
    return results, outputs


def make_classification_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, consistently sorted model comparison table."""
    required = {"Model", "Accuracy", "Weighted F1", "Macro F1"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    return results.sort_values(
        ["Weighted F1", "Accuracy"], ascending=False
    ).reset_index(drop=True)
