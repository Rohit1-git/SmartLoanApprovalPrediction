"""Data cleaning, model training, charts and prediction for SmartLoan."""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.conf import settings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
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

TARGET = "Loan_Status"
MODEL_PATH = Path(settings.MEDIA_ROOT) / "loan_model.joblib"
DATA_PATH = Path(settings.MEDIA_ROOT) / "active_dataset.csv"
META_PATH = Path(settings.MEDIA_ROOT) / "model_metadata.json"
MIN_ROWS = 500  # College requirement: use at least 500 real-world records.
MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 100


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric outliers using IQR limits learned from training data only."""

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IQRClipper expects a pandas DataFrame.")
        self.bounds_ = {}
        self.numeric_columns_ = X.select_dtypes(include=np.number).columns.tolist()
        for column in self.numeric_columns_:
            values = pd.to_numeric(X[column], errors="coerce")
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            if pd.notna(q1) and pd.notna(q3):
                iqr = q3 - q1
                if iqr > 0:
                    self.bounds_[column] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        return self

    def transform(self, X):
        result = X.copy()
        for column, (lower, upper) in self.bounds_.items():
            if column in result.columns:
                result[column] = pd.to_numeric(result[column], errors="coerce").clip(lower, upper)
        return result


def _paths():
    Path(settings.MEDIA_ROOT).mkdir(parents=True, exist_ok=True)
    return Path(settings.MEDIA_ROOT)


def read_dataset(uploaded_file):
    """Read a CSV/XLSX upload with basic safety and structure checks."""
    filename = str(getattr(uploaded_file, "name", "")).lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise ValueError("Please upload a CSV (.csv) or Excel (.xlsx) file.")
    if getattr(uploaded_file, "size", 0) > MAX_BYTES:
        raise ValueError("The file is larger than 10 MB. Please upload a smaller dataset.")

    try:
        uploaded_file.seek(0)
        if filename.endswith(".csv"):
            try:
                frame = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                frame = pd.read_csv(uploaded_file, encoding="latin-1")
        else:
            frame = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"The file could not be read. Check that it is a valid CSV/XLSX file. Details: {exc}") from exc

    if frame.empty:
        raise ValueError("The uploaded dataset has no rows.")
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError("For this beginner project, the limit is 100,000 rows and 100 columns.")

    frame.columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in frame.columns):
        raise ValueError("One or more columns have an empty name. Rename those columns and upload again.")
    if len(set(column.casefold() for column in frame.columns)) != len(frame.columns):
        raise ValueError("Duplicate column names were found. Please make every column name unique.")

    # Accept Loan_Status with different letter casing, then standardize it.
    target_matches = [column for column in frame.columns if column.casefold() == TARGET.casefold()]
    if target_matches:
        frame = frame.rename(columns={target_matches[0]: TARGET})
    return frame


def normalize_target(series: pd.Series) -> pd.Series:
    """Convert common binary approval labels into Y and N."""
    mapping = {
        "Y": "Y", "YES": "Y", "APPROVED": "Y", "APPROVE": "Y",
        "1": "Y", "1.0": "Y", "TRUE": "Y",
        "N": "N", "NO": "N", "REJECTED": "N", "REJECT": "N",
        "0": "N", "0.0": "N", "FALSE": "N",
    }
    normalized = series.astype("string").str.strip().str.upper()
    return normalized.map(mapping)


def _prepare_features(frame: pd.DataFrame):
    """Remove IDs/empty/constant columns and identify numeric/categorical inputs."""
    data = frame.copy()
    data[TARGET] = normalize_target(data[TARGET])
    invalid_target_rows = int(data[TARGET].isna().sum())
    data = data.dropna(subset=[TARGET]).copy()

    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    if data[TARGET].nunique() != 2:
        raise ValueError("Loan_Status must contain exactly two classes: approved and rejected (for example Y and N).")

    original_feature_columns = [column for column in data.columns if column != TARGET]
    dropped_columns = []

    # Remove common identifiers because unique IDs normally do not help prediction.
    for column in list(original_feature_columns):
        normalized_name = re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_")
        if normalized_name in {"id", "loan_id", "customer_id", "applicant_id", "application_id", "record_id"} or normalized_name.endswith("_id"):
            dropped_columns.append(column)
            data = data.drop(columns=[column])

    # Drop entirely empty columns.
    for column in list(data.columns):
        if column != TARGET and data[column].isna().all():
            dropped_columns.append(column)
            data = data.drop(columns=[column])

    X = data.drop(columns=[TARGET]).copy()
    y = data[TARGET].copy()

    # Convert numeric-looking text columns; labels such as Dependents='3+' remain categorical.
    for column in X.columns:
        if X[column].dtype == "object" or str(X[column].dtype).startswith("string"):
            text_values = X[column].astype("string").str.strip()
            converted = pd.to_numeric(text_values, errors="coerce")
            non_empty = int(text_values.notna().sum())
            if non_empty and (converted.notna().sum() / non_empty) >= 0.90:
                X[column] = converted
            else:
                text_values = text_values.replace({"": pd.NA})
                X[column] = text_values.astype(object).where(text_values.notna(), np.nan)

    # Remove constant columns after type conversion.
    for column in list(X.columns):
        if X[column].nunique(dropna=True) <= 1:
            dropped_columns.append(column)
            X = X.drop(columns=[column])
            data = data.drop(columns=[column])

    # Replace positive/negative infinity with missing values before imputation.
    X = X.replace([np.inf, -np.inf], np.nan)
    data = data.replace([np.inf, -np.inf], np.nan)

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    if not numeric_columns:
        raise ValueError("The dataset must contain at least one numerical feature, such as ApplicantIncome or LoanAmount.")
    if not categorical_columns:
        raise ValueError("The dataset must contain at least one categorical feature, such as Gender, Education or Property_Area.")
    if X.empty:
        raise ValueError("No usable input features remain after removing ID and constant columns.")

    return data, X, y, numeric_columns, categorical_columns, duplicate_count, invalid_target_rows, dropped_columns


def _figure_base64(fig):
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_charts(frame: pd.DataFrame):
    """Create the four required EDA charts plus a target distribution chart."""
    charts = {}
    plot_frame = frame.copy()
    if len(plot_frame) > 2500:
        plot_frame = plot_frame.sample(2500, random_state=42)
    numeric = plot_frame.select_dtypes(include=[np.number])

    # Target class distribution
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    plot_frame[TARGET].astype(str).value_counts().reindex(["Y", "N"]).dropna().plot(kind="bar", ax=ax)
    ax.set_title("Loan Approval Status Distribution")
    ax.set_xlabel("Loan status (Y = Approved, N = Rejected)")
    ax.set_ylabel("Number of applications")
    charts["target"] = _figure_base64(fig)

    if numeric.shape[1] >= 1:
        first_numeric = numeric.columns[0]
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        pd.to_numeric(plot_frame[first_numeric], errors="coerce").dropna().plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Distribution of {first_numeric}")
        ax.set_xlabel(first_numeric)
        ax.set_ylabel("Frequency")
        charts["hist"] = _figure_base64(fig)

        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        sns.boxplot(x=pd.to_numeric(plot_frame[first_numeric], errors="coerce"), ax=ax)
        ax.set_title(f"Box Plot of {first_numeric}")
        ax.set_xlabel(first_numeric)
        charts["box"] = _figure_base64(fig)

    # Include a numeric version of the target so the heatmap always shows the outcome.
    corr_data = numeric.copy()
    corr_data["Loan_Status_Encoded"] = plot_frame[TARGET].map({"N": 0, "Y": 1}).astype(float)
    corr_data = corr_data.select_dtypes(include=[np.number])
    corr_data = corr_data.loc[:, corr_data.nunique(dropna=True) > 1]
    if corr_data.shape[1] >= 2:
        # Keep the chart readable on wide datasets.
        corr_data = corr_data.iloc[:, :9]
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        sns.heatmap(corr_data.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        charts["heatmap"] = _figure_base64(fig)

    # Scatter plot: first two numeric columns, or first numeric column vs target.
    if numeric.shape[1] >= 2:
        x_col, y_col = numeric.columns[:2]
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        sns.scatterplot(data=plot_frame, x=x_col, y=y_col, hue=TARGET, alpha=0.7, ax=ax)
        ax.set_title(f"{x_col} vs {y_col}")
        charts["scatter"] = _figure_base64(fig)
    elif numeric.shape[1] == 1:
        x_col = numeric.columns[0]
        scatter_frame = plot_frame.copy()
        scatter_frame["Loan_Status_Encoded"] = scatter_frame[TARGET].map({"N": 0, "Y": 1})
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        sns.stripplot(data=scatter_frame, x="Loan_Status_Encoded", y=x_col, hue=TARGET, jitter=0.25, ax=ax)
        ax.set_title(f"{x_col} by Loan Status")
        ax.set_xlabel("Loan status (0 = Rejected, 1 = Approved)")
        charts["scatter"] = _figure_base64(fig)

    return charts


def train_dataset(uploaded_frame: pd.DataFrame):
    """Validate, preprocess, train, evaluate and save the active model."""
    if TARGET not in uploaded_frame.columns:
        raise ValueError(
            "Required target column 'Loan_Status' was not found. "
            "Rename the result/approval column to Loan_Status. Supported values: Y/N, Yes/No, Approved/Rejected, or 1/0."
        )

    original_rows = int(len(uploaded_frame))
    if original_rows < MIN_ROWS:
        raise ValueError(f"The college requirement is at least {MIN_ROWS} records. Your file has {original_rows} rows.")

    cleaned, X, y, numeric_columns, categorical_columns, duplicate_count, invalid_target_rows, dropped_columns = _prepare_features(uploaded_frame)
    if len(cleaned) < MIN_ROWS:
        raise ValueError(
            f"Only {len(cleaned)} valid unique records remain after cleaning. "
            f"At least {MIN_ROWS} are required. Check duplicates and Loan_Status values."
        )
    if y.value_counts().min() < 2:
        raise ValueError("Both Loan_Status classes need at least two records for training and testing.")

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    model = Pipeline(steps=[
        ("iqr_outlier_treatment", IQRClipper()),
        ("preprocessor", preprocessor),
        ("feature_selection", SelectPercentile(score_func=f_classif, percentile=80)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])

    stratify_values = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify_values
    )
    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    except Exception as exc:
        raise ValueError(f"Model training failed. Please check your dataset columns and values. Details: {exc}") from exc

    labels = ["N", "Y"]
    matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)) * 100, 2),
        "precision": round(float(precision_score(y_test, predictions, pos_label="Y", zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, predictions, pos_label="Y", zero_division=0)) * 100, 2),
        "f1": round(float(f1_score(y_test, predictions, pos_label="Y", zero_division=0)) * 100, 2),
        "confusion_matrix": matrix,
        "confusion_labels": labels,
        "test_records": int(len(y_test)),
    }

    # Store a readable, cleaned dataset (missing feature values are kept for summary;
    # the sklearn pipeline imputes them during training and prediction).
    charts = make_charts(cleaned)
    feature_names = list(X.columns)
    category_choices = {}
    for column in categorical_columns:
        values = cleaned[column].dropna().astype(str).unique().tolist()
        category_choices[column] = sorted(values)[:50]

    try:
        transformed_feature_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
        selection_mask = model.named_steps["feature_selection"].get_support()
        selected_feature_names = [name for name, selected in zip(transformed_feature_names, selection_mask) if selected]
    except Exception:
        selected_feature_names = []

    metadata = {
        "features": feature_names,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "categorical_choices": category_choices,
        "selected_features": selected_feature_names,
        "selected_feature_count": len(selected_feature_names),
        "metrics": metrics,
        "rows": int(len(cleaned)),
        "columns": int(len(cleaned.columns)),
        "original_rows": original_rows,
        "duplicates_removed": duplicate_count,
        "invalid_target_rows_removed": invalid_target_rows,
        "dropped_columns": dropped_columns,
        "missing_total": int(cleaned.isna().sum().sum()),
        "missing": {str(column): int(count) for column, count in cleaned.isna().sum().items()},
        "class_counts": {str(key): int(value) for key, value in y.value_counts().items()},
        "target": TARGET,
        "model_name": "Logistic Regression",
        "train_records": int(len(y_train)),
        "test_records": int(len(y_test)),
    }

    media_dir = _paths()
    temp_model = media_dir / "loan_model.tmp.joblib"
    temp_data = media_dir / "active_dataset.tmp.csv"
    temp_meta = media_dir / "model_metadata.tmp.json"
    try:
        joblib.dump(model, temp_model)
        cleaned.to_csv(temp_data, index=False)
        temp_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temp_model.replace(MODEL_PATH)
        temp_data.replace(DATA_PATH)
        temp_meta.replace(META_PATH)
    finally:
        for temporary_file in (temp_model, temp_data, temp_meta):
            if temporary_file.exists():
                temporary_file.unlink()

    return metadata, cleaned, charts


def get_metadata():
    if not META_PATH.exists():
        return None
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_charts():
    if not DATA_PATH.exists():
        return {}
    try:
        return make_charts(pd.read_csv(DATA_PATH))
    except Exception:
        return {}


def predict_one(posted_values):
    """Return the prediction and model confidence for one applicant."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise ValueError("Please upload and train a dataset before making a prediction.")

    metadata = get_metadata()
    if not metadata:
        raise ValueError("Model information is missing. Please upload the dataset and train the model again.")

    row = {}
    for column in metadata["features"]:
        raw_value = posted_values.get(column, "")
        if column in metadata["numeric_features"]:
            if str(raw_value).strip() == "":
                row[column] = np.nan
            else:
                try:
                    row[column] = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"'{column}' must contain a valid number.") from exc
        else:
            row[column] = str(raw_value).strip() if str(raw_value).strip() else np.nan

    model = joblib.load(MODEL_PATH)
    input_frame = pd.DataFrame([row], columns=metadata["features"])
    prediction = str(model.predict(input_frame)[0])
    probability_by_class = {}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_frame)[0]
        class_names = model.named_steps["classifier"].classes_
        probability_by_class = {str(label): float(probability) for label, probability in zip(class_names, probabilities)}

    approved_probability = round(probability_by_class.get("Y", 0.0) * 100, 2)
    rejected_probability = round(probability_by_class.get("N", 0.0) * 100, 2)
    confidence = round(max(probability_by_class.values()) * 100, 2) if probability_by_class else None

    return {
        "label": "Approved" if prediction == "Y" else "Rejected",
        "approved_probability": approved_probability,
        "rejected_probability": rejected_probability,
        "confidence": confidence,
    }
