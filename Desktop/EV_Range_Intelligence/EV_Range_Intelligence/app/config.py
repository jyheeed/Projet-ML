from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
APP_NAME = "EV Range Intelligence API"
APP_VERSION = "2.0.0"
VEHICLE_NAME = "BMW i3 (60 Ah)"
USABLE_BATTERY_KWH = 18.8
HIGH_CONSUMPTION_RATE = 0.143
LOW_CONSUMPTION_RATE = 0.107

MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "app" / "static"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "best_model_rf.pkl"))
FEATURE_NAMES_PATH = Path(os.getenv("FEATURE_NAMES_PATH", MODELS_DIR / "feature_names.pkl"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
