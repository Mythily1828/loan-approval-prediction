"""Train and evaluate loan-approval classifiers from a CSV dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.api.types import is_integer_dtype, is_numeric_dtype


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "loan_model.joblib"
DEFAULT_REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42


def make_demo_data(rows: int = 614, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create clearly synthetic data for testing the pipeline without a CSV."""
    if rows < 10:
        raise ValueError("Demo data must contain at least 10 rows.")

    rng = np.random.default_rng(random_state)
    data = pd.DataFrame(
        {
            "Loan_ID": [f"DEMO-{i:04d}" for i in range(rows)],
            "Gender": rng.choice(["Male", "Female"], rows, p=[0.81, 0.19]),
            "Married": rng.choice(["Yes", "No"], rows, p=[0.65, 0.35]),
            "Dependents": rng.choice(["0", "1", "2", "3+"], rows),
            "Education": rng.choice(["Graduate", "Not Graduate"], rows, p=[0.78, 0.22]),
            "Self_Employed": rng.choice(["No", "Yes"], rows, p=[0.86, 0.14]),
            "ApplicantIncome": rng.lognormal(8.6, 0.55, rows).round(),
            "CoapplicantIncome": np.where(
                rng.random(rows) < 0.4, 0, rng.lognormal(7.4, 0.8, rows)
            ).round(),
            "Loan_Amount_Term": rng.choice([360, 180, 480, 300, 240], rows),
            "Credit_History": rng.choice([1.0, 0.0], rows, p=[0.84, 0.16]),
            "Property_Area": rng.choice(["Semiurban", "Urban", "Rural"], rows),
        }
    )
    data["LoanAmount"] = (
        (data["ApplicantIncome"] + data["CoapplicantIncome"])
        / 38
        * rng.lognormal(0, 0.3, rows)
    ).round()
    score = (
        3.2 * data["Credit_History"]
        + 0.5 * (data["Property_Area"] == "Semiurban")
        + 0.3 * (data["Married"] == "Yes")
        + 0.2 * (data["Education"] == "Graduate")
        - 0.004 * data["LoanAmount"]
        + 0.00006 * data["ApplicantIncome"]
        + rng.normal(0, 0.9, rows)
        - 1.6
    )
    data["Loan_Status"] = np.where(score > 0.7, "Y", "N")

    for column, fraction in (
        ("Gender", 0.02),
        ("Married", 0.005),
        ("Dependents", 0.025),
        ("Self_Employed", 0.05),
        ("LoanAmount", 0.035),
        ("Loan_Amount_Term", 0.02),
        ("Credit_History", 0.08),
    ):
        data.loc[rng.random(rows) < fraction, column] = np.nan
    return data


def infer_approved_label(labels: list[Any], requested: str | None = None) -> Any:
    """Resolve the class considered approved, requiring an explicit choice if unclear."""
    if requested is not None:
        matches = [label for label in labels if str(label) == requested]
        if len(matches) != 1:
            raise ValueError(
                f"Approved label {requested!r} does not match a target value. "
                f"Available values: {[str(label) for label in labels]}"
            )
        return matches[0]

    approved_aliases = {"y", "yes", "approved", "approve", "1", "true", "positive"}
    matches = [
        label for label in labels if str(label).strip().casefold() in approved_aliases
    ]
    if len(matches) != 1:
        raise ValueError(
            "Could not identify the approved target class. "
            "Pass --approved-label with the exact value used in the CSV. "
            f"Available values: {[str(label) for label in labels]}"
        )
    return matches[0]


def _clean_category(value: Any) -> Any:
    return str(value).strip() if pd.notna(value) else np.nan


def _prepare_training_frame(
    data: pd.DataFrame, target_column: str, approved_label: str | None
) -> tuple[pd.DataFrame, pd.Series, Any, Any]:
    if data.empty:
        raise ValueError("The training dataset is empty.")
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    target_column = target_column.strip()
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Column names must be unique; duplicates: {duplicates}")
    if target_column not in data.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found. "
            f"Available columns: {list(data.columns)}"
        )

    usable = data.dropna(subset=[target_column]).copy()
    y = usable.pop(target_column)
    y = y.map(lambda value: value.strip() if isinstance(value, str) else value)
    if y.nunique() != 2:
        raise ValueError(
            f"Target column {target_column!r} must have exactly two non-missing "
            f"classes; found {y.nunique()}."
        )
    labels = list(pd.unique(y))
    positive_label = infer_approved_label(labels, approved_label)
    negative_label = next(label for label in labels if label != positive_label)

    id_columns = [
        column
        for column in usable.columns
        if str(column).strip().casefold() in {"id", "loan_id", "applicant_id"}
    ]
    if id_columns:
        usable = usable.drop(columns=id_columns)
    if usable.shape[1] == 0:
        raise ValueError("No predictor columns remain after excluding the target and ID.")
    empty_features = usable.columns[usable.isna().all()].tolist()
    if empty_features:
        raise ValueError(
            "These predictor columns contain no values and cannot be trained: "
            f"{empty_features}"
        )

    usable.columns = usable.columns.map(str)
    return usable, y, positive_label, negative_label


def _engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    engineered = features.copy()
    income_columns = ("ApplicantIncome", "CoapplicantIncome")
    if all(column in engineered.columns for column in income_columns):
        applicant_income = pd.to_numeric(
            engineered["ApplicantIncome"], errors="coerce"
        )
        coapplicant_income = pd.to_numeric(
            engineered["CoapplicantIncome"], errors="coerce"
        )
        total_income = applicant_income + coapplicant_income
        if (total_income.dropna() < 0).any():
            raise ValueError("Income fields must not contain negative values.")
        if "TotalIncome_log" in engineered.columns:
            raise ValueError(
                "Input already contains 'TotalIncome_log'; rename the column to avoid "
                "conflicting with the engineered feature."
            )
        engineered["TotalIncome_log"] = np.log1p(total_income)
        engineered = engineered.drop(columns=list(income_columns))

    if "LoanAmount" in engineered.columns:
        loan_amount = pd.to_numeric(engineered["LoanAmount"], errors="coerce")
        if (loan_amount.dropna() < 0).any():
            raise ValueError("LoanAmount must not contain negative values.")
        if "LoanAmount_log" in engineered.columns:
            raise ValueError(
                "Input already contains 'LoanAmount_log'; rename the column to avoid "
                "conflicting with the engineered feature."
            )
        engineered["LoanAmount_log"] = np.log1p(loan_amount)
        engineered = engineered.drop(columns=["LoanAmount"])
    return engineered


def _make_pipeline(
    features: pd.DataFrame, classifier: Any
) -> tuple[Pipeline, list[str], list[str]]:
    numeric_columns = [
        column for column in features.columns if is_numeric_dtype(features[column])
    ]
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessing = ColumnTransformer(transformers, remainder="drop")
    return (
        Pipeline([("preprocessing", preprocessing), ("classifier", classifier)]),
        numeric_columns,
        categorical_columns,
    )


def _feature_schema(
    features: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]
) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for column in features.columns:
        series = features[column]
        if column in numeric_columns:
            default = float(pd.to_numeric(series, errors="coerce").median())
            schema.append(
                {
                    "name": column,
                    "kind": "numeric",
                    "default": default,
                    "step": 1.0 if is_integer_dtype(series) else 0.01,
                }
            )
        else:
            values = sorted(
                {_clean_category(value) for value in series.dropna().tolist()},
                key=str.casefold,
            )
            default = (
                _clean_category(series.mode(dropna=True).iloc[0])
                if not series.mode().empty
                else ""
            )
            schema.append(
                {
                    "name": column,
                    "kind": "categorical",
                    "choices": values,
                    "default": default,
                }
            )
    return schema


def _write_reports(
    metrics: pd.DataFrame,
    actual: pd.Series,
    predicted: np.ndarray,
    negative_label: Any,
    positive_label: Any,
    pipeline: Pipeline,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(metrics["Model"], metrics["Accuracy"], color="#3b82f6")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Holdout accuracy by model")
    axes[0].tick_params(axis="x", rotation=20)
    for index, accuracy in enumerate(metrics["Accuracy"]):
        axes[0].text(index, accuracy + 0.02, f"{accuracy:.1%}", ha="center")

    matrix = confusion_matrix(
        actual, predicted, labels=[negative_label, positive_label]
    )
    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_title("Best model confusion matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_xticks([0, 1], [str(negative_label), str(positive_label)])
    axes[1].set_yticks([0, 1], [str(negative_label), str(positive_label)])
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation.png", dpi=160)
    plt.close(figure)

    preprocessing = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
    else:
        importance_values = np.abs(classifier.coef_).ravel()
    feature_names = preprocessing.get_feature_names_out()
    importance = pd.Series(importance_values, index=feature_names).nlargest(12).sort_values()
    figure, axis = plt.subplots(figsize=(8, max(3.5, 0.36 * len(importance))))
    axis.barh(importance.index, importance.values, color="#2563eb")
    axis.set_title(f"Top features — {pipeline.named_steps['classifier'].__class__.__name__}")
    axis.set_xlabel("Importance (absolute coefficient or tree importance)")
    figure.tight_layout()
    figure.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(figure)

def train_and_evaluate(
    data: pd.DataFrame,
    target_column: str = "Loan_Status",
    approved_label: str | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> dict[str, Any]:
    """Fit three classifiers, evaluate them on a stratified holdout, and save the winner."""
    input_features, target, positive_label, negative_label = _prepare_training_frame(
        data, target_column, approved_label
    )
    features = _engineer_features(input_features)
    class_counts = target.value_counts()
    if class_counts.min() < 2 or math.ceil(len(target) * 0.2) < 2:
        raise ValueError(
            "At least two examples per class and two holdout rows are required "
            "for a stratified 80/20 split."
        )

    numeric_columns = [
        column for column in features.columns if is_numeric_dtype(features[column])
    ]
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    input_numeric_columns = [
        column for column in input_features.columns if is_numeric_dtype(input_features[column])
    ]
    input_categorical_columns = [
        column for column in input_features.columns if column not in input_numeric_columns
    ]
    for column in categorical_columns:
        features[column] = features[column].map(_clean_category)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        ),
    }

    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    fitted_pipelines: dict[str, Pipeline] = {}
    for name, classifier in candidates.items():
        pipeline, _, _ = _make_pipeline(features, classifier)
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        predictions[name] = prediction
        fitted_pipelines[name] = pipeline
        is_positive = y_test == positive_label
        predicted_positive = prediction == positive_label
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Precision": precision_score(
                    is_positive, predicted_positive, zero_division=0
                ),
                "Recall": recall_score(
                    is_positive, predicted_positive, zero_division=0
                ),
                "F1": f1_score(is_positive, predicted_positive, zero_division=0),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(
        ["Accuracy", "Model"], ascending=[False, True], ignore_index=True
    )
    best_name = str(metrics.iloc[0]["Model"])
    best_pipeline = fitted_pipelines[best_name]
    schema = _feature_schema(
        input_features, input_numeric_columns, input_categorical_columns
    )
    bundle = {
        "pipeline": best_pipeline,
        "feature_columns": list(features.columns),
        "input_columns": list(input_features.columns),
        "feature_schema": schema,
        "target_column": target_column,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "selected_model": best_name,
        "test_metrics": metrics.to_dict(orient="records"),
        "random_state": RANDOM_STATE,
    }

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_file)
    _write_reports(
        metrics,
        y_test,
        predictions[best_name],
        negative_label,
        positive_label,
        best_pipeline,
        Path(reports_dir),
    )
    print(metrics.to_string(index=False, formatters={c: "{:.3f}".format for c in metrics.columns if c != "Model"}))
    print(f"Selected model: {best_name}")
    print(f"Saved model: {model_file}")
    print(f"Saved reports: {Path(reports_dir)}")
    return bundle


def predict_approval(
    bundle: dict[str, Any], applicant: dict[str, Any]
) -> tuple[Any, float]:
    """Return the model's predicted class and probability of the approved class."""
    required = bundle["input_columns"]
    missing = [column for column in required if column not in applicant]
    if missing:
        raise ValueError(f"Missing predictor values: {missing}")
    row = pd.DataFrame([{column: applicant[column] for column in required}])
    row = _engineer_features(row)
    for field in bundle["feature_schema"]:
        if field["kind"] == "categorical":
            column = field["name"]
            if column not in row.columns:
                continue
            row[column] = row[column].map(_clean_category)
    row = row.reindex(columns=bundle["feature_columns"])

    pipeline: Pipeline = bundle["pipeline"]
    predicted = pipeline.predict(row)[0]
    probabilities = pipeline.predict_proba(row)[0]
    classes = list(pipeline.classes_)
    positive_index = next(
        (index for index, label in enumerate(classes) if label == bundle["positive_label"]),
        None,
    )
    if positive_index is None:
        raise ValueError("Saved model does not contain the approved class.")
    return predicted, float(probabilities[positive_index])


def explain_approval_sensitivity(
    bundle: dict[str, Any], applicant: dict[str, Any]
) -> pd.DataFrame:
    """Compare each entered value with its training median/mode, one field at a time."""
    _, current_score = predict_approval(bundle, applicant)
    comparisons = []
    for field in bundle["feature_schema"]:
        name = field["name"]
        typical_value = field["default"]
        alternate_applicant = dict(applicant)
        alternate_applicant[name] = typical_value
        _, typical_score = predict_approval(bundle, alternate_applicant)
        comparisons.append(
            {
                "Field": name.replace("_", " "),
                "Entered value": str(applicant[name]),
                "Typical training value": str(typical_value),
                "Score change vs typical (percentage points)": round(
                    (current_score - typical_score) * 100, 1
                ),
            }
        )
    return pd.DataFrame(comparisons).sort_values(
        "Score change vs typical (percentage points)",
        key=lambda changes: changes.abs(),
        ascending=False,
        ignore_index=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train loan approval models and save the best holdout performer."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, help="Path to the training CSV.")
    source.add_argument(
        "--demo",
        action="store_true",
        help="Train on generated synthetic data for a pipeline demonstration only.",
    )
    parser.add_argument(
        "--target",
        default="Loan_Status",
        help="Name of the binary target column (default: Loan_Status).",
    )
    parser.add_argument(
        "--approved-label",
        help="Exact target value representing approval if it cannot be inferred.",
    )
    parser.add_argument(
        "--model-output", type=Path, default=DEFAULT_MODEL_PATH, help="Model artifact path."
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Evaluation output folder."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.demo:
        training_data = make_demo_data()
        print("Using generated synthetic data; these results are not real-world evidence.")
    elif args.data is not None:
        if not args.data.is_file():
            raise FileNotFoundError(f"Training CSV does not exist: {args.data}")
        training_data = pd.read_csv(args.data)
    else:
        raise SystemExit("Provide --data PATH or explicitly opt in to --demo.")

    train_and_evaluate(
        training_data,
        target_column=args.target,
        approved_label=args.approved_label,
        model_path=args.model_output,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
