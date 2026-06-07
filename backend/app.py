from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml.trainer import get_best_model_name, load_model, train_all
from ml.visualizer import FIGURES_DIR, generate_all_figures

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

training_cache: dict | None = None
figures_list: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global training_cache, figures_list
    training_cache = train_all()
    figures_list = generate_all_figures(training_cache)
    yield


app = FastAPI(
    title="CCPP Power Output Predictor",
    description="Predict full load electrical power output of a combined cycle power plant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    AT: float = Field(..., ge=1.81, le=37.11, description="Ambient Temperature (°C)")
    V: float = Field(..., ge=25.36, le=81.56, description="Exhaust Vacuum (cm Hg)")
    AP: float = Field(..., ge=992.89, le=1033.30, description="Atmospheric Pressure (mbar)")
    RH: float = Field(..., ge=25.56, le=100.16, description="Relative Humidity (%)")
    model: str | None = Field(None, description="Model name; defaults to best model")


class PredictionOutput(BaseModel):
    predicted_pe: float
    model_used: str
    unit: str = "MW"


@app.get("/api/health")
def health():
    return {"status": "ok", "models_trained": training_cache is not None}


@app.get("/api/models")
def list_models():
    if not training_cache:
        raise HTTPException(503, "Models not trained yet")
    return {
        "best_model": training_cache["best_model"],
        "models": list(training_cache["results"].keys()),
    }


@app.get("/api/metrics")
def get_metrics():
    if not training_cache:
        raise HTTPException(503, "Models not trained yet")
    return {
        "train_samples": training_cache["train_samples"],
        "test_samples": training_cache["test_samples"],
        "train_sheets": training_cache["train_sheets"],
        "test_sheets": training_cache["test_sheets"],
        "best_model": training_cache["best_model"],
        "results": training_cache["results"],
        "feature_ranges": training_cache["feature_ranges"],
    }


@app.get("/api/figures")
def list_figures():
    return {"figures": figures_list}


@app.get("/api/figures/{filename}")
def get_figure(filename: str):
    path = FIGURES_DIR / filename
    if not path.exists() or path.suffix != ".png":
        raise HTTPException(404, "Figure not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    model_name = payload.model or get_best_model_name()
    try:
        model = load_model(model_name)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{model_name}' not found")

    import pandas as pd

    features = pd.DataFrame([{
        "AT": payload.AT,
        "V": payload.V,
        "AP": payload.AP,
        "RH": payload.RH,
    }])
    prediction = float(model.predict(features)[0])
    return PredictionOutput(predicted_pe=round(prediction, 2), model_used=model_name)


@app.post("/api/retrain")
def retrain():
    global training_cache, figures_list
    training_cache = train_all()
    figures_list = generate_all_figures(training_cache)
    return {"message": "Retraining complete", "best_model": training_cache["best_model"]}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
