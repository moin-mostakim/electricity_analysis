#!/usr/bin/env python3
"""Train models and generate figures without starting the web server."""

from ml.trainer import train_all
from ml.visualizer import generate_all_figures

if __name__ == "__main__":
    print("Training models on Sheet1-3, testing on Sheet4-5...")
    output = train_all()
    print(f"\nBest model: {output['best_model']}\n")
    for name, r in output["results"].items():
        print(f"  {name:20s}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}")
    print("\nGenerating figures...")
    figures = generate_all_figures(output)
    print("Saved:", ", ".join(figures))
    print("\nDone. Run ./run.sh to start the web dashboard.")
