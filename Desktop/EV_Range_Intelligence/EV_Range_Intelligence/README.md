# EV Range Intelligence Pro

A refactored and deployment-ready version of the original BMW i3 EV consumption project.

## What changed

- Rebuilt the backend into a real `app/` package.
- Centralized feature construction on the server side.
- Removed duplicated model artifacts and fragile relative paths.
- Added a cleaner Streamlit chatbot that calls the new API contract.
- Added tests, environment configuration, and better project structure.
- Preserved the original notebook, raw data, MLflow tracking, and Power BI assets.

## New structure

```text
EV_Range_Intelligence_Pro/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── services/
│   └── static/
├── chatbot/
├── models/
├── data/
├── dashboard/
├── notebooks/
├── mlflow/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open the web UI at `http://localhost:8000`.

On Windows, you can also use:

- `scripts/run_api.bat` or `scripts/run_api.ps1`
- `scripts/run_chatbot.bat` or `scripts/run_chatbot.ps1`

## Run the chatbot

```bash
streamlit run chatbot/chatbot.py
```

## Run tests

```bash
pytest
```

## API endpoints

- `GET /health`
- `POST /api/v1/predict`
  - accepts high-level trip inputs: distance, duration, SoC, temperature, drive style, terrain
- `POST /predict`
  - backward-compatible endpoint that accepts either the compact scenario input or the old raw feature payload
- `POST /api/v1/predict/raw`
  - accepts the full 31-feature payload for debugging or notebook interoperability

## Why this is better

The old project repeated the trip-to-feature logic in the frontend and chatbot. That is a maintenance trap. In this version, the backend owns that logic, which makes the web app, API, and chatbot consistent.

## Limits that still exist

- The ML model is still a binary consumption classifier, not a true continuous range regressor.
- Range is estimated from the prediction class using fixed consumption rates.
- The trained model is still based on one vehicle family and 69 usable trips.

## Next high-value upgrade

Build a second model version that predicts `Energy_per_km` directly through regression, then compare it against the current classifier approach.


## Compatibility note

The trained model artifact was created with scikit-learn 1.7.2. Keep that version pinned when you install dependencies to avoid serialization mismatch drama.
