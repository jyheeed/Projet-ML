from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

app = FastAPI(title="Flight Delay Prediction API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FlightInput(BaseModel):
    crs_elapsed_time: float
    distance: float
    dep_hour: int
    arr_hour: int
    month: int
    day_of_week: int
    day_of_month: int
    is_weekend: int
    season: int
    time_block: int
    is_busy_origin: int
    is_busy_dest: int
    is_evening: int
    is_early: int
    is_long_haul: int
    is_holiday: int
    origin_te: float
    dest_te: float
    airline_te: float
    route_te: float
    airline_hour_te: float
    origin_hour_te: float
    airline_month_te: float
    route_dow_te: float
    dest_hour_te: float

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "healthy", "model": "HGB_PreFlight", "version": "3.0"}

@app.post("/predict")
def predict(flight: FlightInput):
    data = {
        "CRS_ELAPSED_TIME": flight.crs_elapsed_time,
        "DISTANCE": flight.distance,
        "DEP_HOUR": flight.dep_hour,
        "ARR_HOUR": flight.arr_hour,
        "MONTH": flight.month,
        "DAY_OF_WEEK": flight.day_of_week,
        "DAY_OF_MONTH": flight.day_of_month,
        "IS_WEEKEND": flight.is_weekend,
        "SEASON": flight.season,
        "TIME_BLOCK": flight.time_block,
        "IS_BUSY_ORIGIN": flight.is_busy_origin,
        "IS_BUSY_DEST": flight.is_busy_dest,
        "IS_EVENING": flight.is_evening,
        "IS_EARLY": flight.is_early,
        "IS_LONG_HAUL": flight.is_long_haul,
        "IS_HOLIDAY": flight.is_holiday,
        "ORIGIN_TE": flight.origin_te,
        "DEST_TE": flight.dest_te,
        "AIRLINE_TE": flight.airline_te,
        "ROUTE_TE": flight.route_te,
        "AIRLINE_HOUR_TE": flight.airline_hour_te,
        "ORIGIN_HOUR_TE": flight.origin_hour_te,
        "AIRLINE_MONTH_TE": flight.airline_month_te,
        "ROUTE_DOW_TE": flight.route_dow_te,
        "DEST_HOUR_TE": flight.dest_hour_te,
    }
    df = pd.DataFrame([data])[feature_names]
    numeric_features = ["CRS_ELAPSED_TIME", "DISTANCE"]
    df[numeric_features] = scaler.transform(df[numeric_features])
    prediction = model.predict(df.values)[0]
    probability = model.predict_proba(df.values)[0]
    return {
        "prediction": "DELAYED" if prediction == 1 else "ON TIME",
        "delay_probability": round(float(probability[1]), 4),
        "on_time_probability": round(float(probability[0]), 4),
    }

@app.post("/predict/batch")
def predict_batch(flights: list[FlightInput]):
    results = [predict(f) for f in flights]
    return {"predictions": results, "count": len(results)}
