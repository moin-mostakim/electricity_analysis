from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .data_loader import FEATURES, TARGET, load_fold_data
from .trainer import ModelResult, train_all

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def generate_all_figures(training_output: dict | None = None) -> list[str]:
    if training_output is None:
        training_output = train_all()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    generated.append(_plot_model_comparison(training_output["results"]))
    generated.append(_plot_actual_vs_predicted(training_output))
    generated.append(_plot_feature_correlation())
    generated.append(_plot_feature_vs_pe())
    generated.append(_plot_residuals(training_output))

    return generated


def _plot_model_comparison(results: dict) -> str:
    names = [r["name"].replace("_", " ").title() for r in results.values()]
    mae = [r["mae"] for r in results.values()]
    rmse = [r["rmse"] for r in results.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, mae, width, label="MAE (MW)", color="#3b82f6")
    ax.bar(x + width / 2, rmse, width, label="RMSE (MW)", color="#f59e0b")
    ax.set_xlabel("Model")
    ax.set_ylabel("Error (MW)")
    ax.set_title("Model Performance Comparison (Test Folds: Sheet4 & Sheet5)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _plot_actual_vs_predicted(training_output: dict) -> str:
    from .trainer import MODELS_DIR, build_models, load_fold_data, split_xy
    import joblib

    _, test_df = load_fold_data()
    x_test, y_test = split_xy(test_df)

    best = training_output["best_model"]
    model = joblib.load(MODELS_DIR / f"{best}.joblib")
    y_pred = model.predict(x_test)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.25, s=8, c="#2563eb")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Ideal prediction")
    ax.set_xlabel("Actual PE (MW)")
    ax.set_ylabel("Predicted PE (MW)")
    ax.set_title(f"Actual vs Predicted — {best.replace('_', ' ').title()}")
    r = training_output["results"][best]
    ax.text(
        0.05, 0.95,
        f"MAE: {r['mae']:.3f} MW\nRMSE: {r['rmse']:.3f} MW\nR²: {r['r2']:.4f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _plot_feature_correlation() -> str:
    train_df, test_df = load_fold_data()
    df = pd.concat([train_df, test_df])
    corr = df[FEATURES + [TARGET]].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["AT (°C)", "V (cm Hg)", "AP (mbar)", "RH (%)", "PE (MW)"]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    path = FIGURES_DIR / "correlation_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _plot_feature_vs_pe() -> str:
    train_df, test_df = load_fold_data()
    df = pd.concat([train_df, test_df])

    labels = {
        "AT": ("Ambient Temperature (°C)", "#ef4444"),
        "V": ("Exhaust Vacuum (cm Hg)", "#8b5cf6"),
        "AP": ("Atmospheric Pressure (mbar)", "#10b981"),
        "RH": ("Relative Humidity (%)", "#06b6d4"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (feat, (label, color)) in zip(axes.flat, labels.items()):
        x = df[feat].values
        y = df[TARGET].values
        ax.scatter(x, y, alpha=0.15, s=6, c=color)
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, lr.predict(xs.reshape(-1, 1)), "k-", linewidth=2)
        ax.set_xlabel(label)
        ax.set_ylabel("Electrical Power Output (MW)")
        ax.grid(alpha=0.3)
    fig.suptitle("Input Features vs Electrical Power Output", fontsize=14, y=1.02)
    fig.tight_layout()

    path = FIGURES_DIR / "features_vs_pe.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_residuals(training_output: dict) -> str:
    from .trainer import MODELS_DIR, load_fold_data, split_xy
    import joblib

    _, test_df = load_fold_data()
    x_test, y_test = split_xy(test_df)
    best = training_output["best_model"]
    model = joblib.load(MODELS_DIR / f"{best}.joblib")
    y_pred = model.predict(x_test)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(residuals, bins=50, color="#6366f1", edgecolor="white")
    axes[0].set_xlabel("Residual (MW)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Distribution")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.25, s=8, c="#6366f1")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted PE (MW)")
    axes[1].set_ylabel("Residual (MW)")
    axes[1].set_title("Residuals vs Predicted Values")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Residual Analysis — {best.replace('_', ' ').title()}", fontsize=13)
    fig.tight_layout()

    path = FIGURES_DIR / "residual_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name
