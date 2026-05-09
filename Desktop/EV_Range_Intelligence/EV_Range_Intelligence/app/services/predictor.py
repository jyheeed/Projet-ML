from __future__ import annotations

from functools import lru_cache

import joblib

from app.config import (
    FEATURE_NAMES_PATH,
    HIGH_CONSUMPTION_RATE,
    LOW_CONSUMPTION_RATE,
    MODEL_PATH,
    USABLE_BATTERY_KWH,
    VEHICLE_NAME,
)
from app.schemas import PredictionResponse
from app.services.feature_builder import ScenarioContext, payload_to_model_dataframe


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[object, list[str]]:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, feature_names


def estimate_range_km(soc_start: float, predicted_label: int) -> tuple[float, float, float]:
    energy_available = (soc_start / 100.0) * USABLE_BATTERY_KWH
    consumption_rate = HIGH_CONSUMPTION_RATE if predicted_label == 1 else LOW_CONSUMPTION_RATE
    estimated_range = energy_available / consumption_rate
    return estimated_range, consumption_rate, energy_available


def build_notes(context: ScenarioContext, prediction_label: int, estimated_range: float) -> list[str]:
    notes: list[str] = []
    if context.climate_power_kw >= 3:
        notes.append("Climate load is significant. Pre-heating or pre-cooling while plugged in would help.")
    elif context.climate_power_kw <= 0.3:
        notes.append("Climate impact is low in this scenario.")

    if context.terrain == "mountain":
        notes.append("Mountain terrain increases energy demand because of elevation gain.")
    if context.drive_style == "highway":
        notes.append("Highway driving usually reduces efficiency because of sustained speed.")
    if prediction_label == 1:
        notes.append("The model expects a higher-than-median consumption profile for this trip.")
    else:
        notes.append("The model expects a relatively efficient trip profile for this trip.")
    notes.append(f"Estimated range is based on the {VEHICLE_NAME} usable battery of {USABLE_BATTERY_KWH} kWh.")
    return notes


def predict_from_payload(payload: dict[str, float], drive_style: str, terrain: str, climate_load_kw: float) -> PredictionResponse:
    model, feature_names = load_artifacts()
    df = payload_to_model_dataframe(payload, feature_names)
    prediction = int(model.predict(df.values)[0])
    probability = model.predict_proba(df.values)[0]
    estimated_range, consumption, energy_available = estimate_range_km(payload["soc_start"], prediction)
    high_probability = round(float(probability[1]), 4)
    low_probability = round(float(probability[0]), 4)
    confidence = round(max(high_probability, low_probability), 4)

    context = ScenarioContext(
        drive_style=drive_style,
        terrain=terrain,
        climate_power_kw=climate_load_kw,
        temperature_gap=payload["temp_diff_mean"],
        energy_used_kwh=(payload["soc_drop"] / 100) * USABLE_BATTERY_KWH,
        soc_end=payload["soc_end"],
        raw_payload=payload,
    )
    notes = build_notes(context, prediction, estimated_range)

    scenario_summary = (
        f"{payload['distance']:.0f} km trip, {payload['ambient_temp']:.0f}°C ambient, "
        f"{payload['soc_start']:.0f}% starting SoC, {drive_style} driving on {terrain} terrain"
    )
    return PredictionResponse(
        prediction="HIGH CONSUMPTION" if prediction == 1 else "LOW CONSUMPTION",
        confidence=confidence,
        high_probability=high_probability,
        low_probability=low_probability,
        estimated_range_km=round(float(estimated_range), 1),
        estimated_consumption_kwh_km=round(float(consumption), 3),
        energy_available_kwh=round(float(energy_available), 1),
        scenario_summary=scenario_summary,
        driving_style=drive_style,
        terrain=terrain,
        climate_load_kw=round(float(climate_load_kw), 2),
        notes=notes,
    )
