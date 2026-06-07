from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .data_loader import FEATURES, TARGET, load_fold_data, split_xy

MODELS_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "saved_models"


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    r2: float
    coefficients: dict[str, float] | None = None
    predictions: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)


def _metrics(y_true, y_pred) -> tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0
    return float(mae), rmse, r2


def build_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
    }


def train_all() -> dict:
    train_df, test_df = load_fold_data()
    x_train, y_train = split_xy(train_df)
    x_test, y_test = split_xy(test_df)

    results: dict[str, ModelResult] = {}
    best_name = ""
    best_rmse = float("inf")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in build_models().items():
        if name == "linear_regression":
            fitted = Pipeline([("scaler", StandardScaler()), ("model", model)])
        else:
            fitted = model

        fitted.fit(x_train, y_train)
        y_pred = fitted.predict(x_test)
        mae, rmse, r2 = _metrics(y_test.values, y_pred)

        coeffs = None
        if name == "linear_regression":
            lr = fitted.named_steps["model"]
            scaler = fitted.named_steps["scaler"]
            raw_coef = lr.coef_ / scaler.scale_
            raw_intercept = lr.intercept_ - np.sum(raw_coef * scaler.mean_)
            coeffs = {"intercept": float(raw_intercept)}
            coeffs.update({f: float(c) for f, c in zip(FEATURES, raw_coef)})

        results[name] = ModelResult(
            name=name,
            mae=mae,
            rmse=rmse,
            r2=r2,
            coefficients=coeffs,
            predictions=y_pred.tolist(),
            actuals=y_test.tolist(),
        )

        joblib.dump(fitted, MODELS_DIR / f"{name}.joblib")

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name

    (MODELS_DIR / "best_model.txt").write_text(best_name)

    return {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_sheets": ["Sheet1", "Sheet2", "Sheet3"],
        "test_sheets": ["Sheet4", "Sheet5"],
        "best_model": best_name,
        "results": {k: _result_to_dict(v) for k, v in results.items()},
        "feature_ranges": _feature_ranges(train_df, test_df),
    }


def _feature_ranges(train_df, test_df) -> dict:
    combined = __import__("pandas").concat([train_df, test_df])
    ranges = {}
    for col in FEATURES + [TARGET]:
        ranges[col] = {
            "min": float(combined[col].min()),
            "max": float(combined[col].max()),
            "mean": float(combined[col].mean()),
        }
    return ranges


def _result_to_dict(r: ModelResult) -> dict:
    return {
        "name": r.name,
        "mae": round(r.mae, 4),
        "rmse": round(r.rmse, 4),
        "r2": round(r.r2, 4),
        "coefficients": r.coefficients,
    }


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model '{name}' not trained yet.")
    return joblib.load(path)


def get_best_model_name() -> str:
    path = MODELS_DIR / "best_model.txt"
    if path.exists():
        return path.read_text().strip()
    return "linear_regression"
