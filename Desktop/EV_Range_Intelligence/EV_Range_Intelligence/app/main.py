from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import ALLOWED_ORIGINS, APP_NAME, APP_VERSION, STATIC_DIR, VEHICLE_NAME
from app.schemas import PredictionResponse, RawFeatureInput, ScenarioInput
from app.services.feature_builder import build_features_from_scenario
from app.services.predictor import predict_from_payload

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_PATH = STATIC_DIR / "index.html"


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "vehicle": VEHICLE_NAME,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_from_scenario(scenario: ScenarioInput) -> PredictionResponse:
    try:
        context = build_features_from_scenario(scenario)
        return predict_from_payload(
            context.raw_payload,
            drive_style=context.drive_style,
            terrain=context.terrain,
            climate_load_kw=context.climate_power_kw,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/predict/raw", response_model=PredictionResponse)
def predict_from_raw(raw: RawFeatureInput) -> PredictionResponse:
    try:
        return predict_from_payload(
            raw.model_dump(),
            drive_style="raw_input",
            terrain="raw_input",
            climate_load_kw=raw.climate_power_mean,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def legacy_predict(payload: dict = Body(...)) -> PredictionResponse:
    """Backward-compatible endpoint for existing clients."""
    try:
        if "velocity_mean" in payload:
            raw = RawFeatureInput(**payload)
            return predict_from_payload(
                raw.model_dump(),
                drive_style="raw_input",
                terrain="raw_input",
                climate_load_kw=raw.climate_power_mean,
            )

        scenario = ScenarioInput(**payload)
        context = build_features_from_scenario(scenario)
        return predict_from_payload(
            context.raw_payload,
            drive_style=context.drive_style,
            terrain=context.terrain,
            climate_load_kw=context.climate_power_kw,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

