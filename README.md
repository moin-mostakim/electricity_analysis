# Combined Cycle Power Plant — Power Output Prediction

Predict full load electrical power output (PE) of a combined cycle power plant using the UCI CCPP dataset.

## Project Overview

- **Dataset**: [UCI Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)
- **Training**: Sheet1, Sheet2, Sheet3 (3 folds)
- **Testing**: Sheet4, Sheet5 (2 folds)
- **Features**: AT (°C), V (cm Hg), AP (mbar), RH (%)
- **Target**: PE (MW) — net hourly electrical energy output
- **Reference**: Tüfekci (2014), *International Journal of Electrical Power & Energy Systems*

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run the server (trains models & generates figures on startup)
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

## Project Structure

```
machine_learning_electricity/
├── CCPP/                    # UCI dataset (Folds5x2_pp.xlsx)
├── backend/
│   ├── app.py               # FastAPI server
│   ├── ml/
│   │   ├── data_loader.py   # Load train/test folds
│   │   ├── trainer.py       # Train & evaluate models
│   │   └── visualizer.py    # Generate result figures
│   ├── figures/             # Generated PNG figures (for submission)
│   └── saved_models/        # Trained model files
└── frontend/
    ├── index.html           # Dashboard UI
    ├── styles.css
    └── app.js
```

## Models

| Model | Description |
|-------|-------------|
| Linear Regression | Multi-variable linear model (Eq. 2 in paper) |
| Ridge Regression | Regularized linear regression |
| Random Forest | Ensemble tree model for comparison |

Best model is selected by lowest RMSE on test folds.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server status |
| GET | `/api/metrics` | Model MAE, RMSE, R² |
| GET | `/api/figures` | List generated figures |
| GET | `/api/figures/{name}` | Download figure PNG |
| POST | `/api/predict` | Predict PE from sensor inputs |

## Deliverables

- **Source code**: This repository
- **Model results (figures)**: Saved in `backend/figures/` and displayed in the web UI
