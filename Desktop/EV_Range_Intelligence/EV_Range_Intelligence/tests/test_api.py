from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint_returns_prediction():
    payload = {
        "distance": 40,
        "duration": 50,
        "soc_start": 70,
        "ambient_temp": -2,
        "cabin_temp": 22,
        "drive_style": "mixed",
        "terrain": "hilly",
    }
    response = client.post("/api/v1/predict", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["prediction"] in {"HIGH CONSUMPTION", "LOW CONSUMPTION"}
    assert data["estimated_range_km"] > 0
    assert data["confidence"] > 0


def test_predict_endpoint_rejects_unrealistic_average_speed():
    payload = {
        "distance": 300,
        "duration": 30,
        "soc_start": 80,
        "ambient_temp": 10,
        "cabin_temp": 22,
        "drive_style": "mixed",
        "terrain": "flat",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
