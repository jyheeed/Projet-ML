<<<<<<< HEAD
# Projet-ML
=======
 ✈️ Flight Delay Prediction — Machine Learning Pipeline

## Project Overview

This project builds a machine learning model to predict whether a US domestic flight will arrive more than 15 minutes late. The model uses only **pre-departure information** — meaning it predicts delays before the plane even leaves the gate. This is the kind of tool an airline, travel app, or passenger could use to anticipate disruptions.

**Dataset:** US Bureau of Transportation Statistics — 3 million domestic flights (2019–2023)

**Target Variable:** `IS_DELAYED` — binary (1 = arrival delay > 15 min, 0 = on time)

**Final Models:** HistGradientBoostingClassifier and RandomForestClassifier

---

## Project Evolution — 3 Iterations

### Iteration 1 — The "Cheating" Model (93% accuracy)

The first version included `DEP_DELAY` (departure delay in minutes) as a feature. This gave spectacular results — 93% accuracy, 0.97 ROC-AUC — but it was misleading. `DEP_DELAY` has a 0.96 correlation with the target, so the model was essentially learning one rule: "if the flight left late, it arrives late." That's not prediction, that's observation.

**What was wrong:**
- `DEP_DELAY` dominated every other feature
- The model ignored airlines, routes, time patterns — everything interesting
- It would be useless in practice because you only know DEP_DELAY after departure

### Iteration 2 — Pre-Flight Attempt (64% accuracy)

The second version removed `DEP_DELAY` and `TAXI_OUT` to force a true pre-flight prediction. However, the replacement features were weak — just basic one-hot encoded airlines and simple historical delay rates computed on the full dataset (including test data). This caused data leakage and inflated the apparent learning while the model was actually memorizing.

**What was wrong:**
- Target encoding leakage: `ROUTE_DELAY_RATE`, `AIRLINE_DELAY_RATE`, etc. were computed on the entire dataset before the train/test split, meaning the model saw test data during training
- Only 32 features, most of which were one-hot airline dummies (low signal)
- Aggressive undersampling (60/40) threw away too much useful on-time data
- Weak hyperparameters: only 300 iterations, max_depth=6, learning_rate=0.1

### Iteration 3 — Final Model (target: 72–78% accuracy)

The final version fixes every issue: no leaky features, rich interaction encodings, proper train-only encoding, and tuned hyperparameters. The accuracy is lower than the cheating model but the predictions are real — the model learns genuine patterns across airlines, airports, routes, time, and their combinations.

**What changed:**
- All target encoding is fitted on train labels only (leakage-safe)
- 5 new interaction features that capture compound patterns (e.g., "Spirit at 7PM in July")
- Binary flags for known delay risk factors (evening, holiday, busy airport)
- Less aggressive undersampling (65/35 instead of 60/40)
- Stronger regularization and early stopping to prevent overfitting

---

## Feature Dictionary

### Schedule Features (known at booking)

| Feature | Type | Description |
|---------|------|-------------|
| `CRS_ELAPSED_TIME` | float | Scheduled flight duration in minutes |
| `DISTANCE` | float | Route distance in miles |

### Temporal Features (known at booking)

| Feature | Type | Description |
|---------|------|-------------|
| `DEP_HOUR` | int (0–23) | Scheduled departure hour |
| `ARR_HOUR` | int (0–23) | Scheduled arrival hour |
| `MONTH` | int (1–12) | Month of flight |
| `DAY_OF_WEEK` | int (0–6) | 0=Monday, 6=Sunday |
| `DAY_OF_MONTH` | int (1–31) | Day of the month |
| `IS_WEEKEND` | binary | 1 if Saturday or Sunday |
| `SEASON` | int (1–4) | 1=Winter, 2=Spring, 3=Summer, 4=Fall |
| `TIME_BLOCK` | int (0–3) | 0=Night (0–6h), 1=Morning (7–12h), 2=Afternoon (13–18h), 3=Evening (19–24h) |

### Binary Risk Flags (engineered)

| Feature | Type | Description |
|---------|------|-------------|
| `IS_BUSY_ORIGIN` | binary | 1 if origin is a top-20 busiest US airport (ATL, DFW, DEN, ORD, LAX, etc.) |
| `IS_BUSY_DEST` | binary | 1 if destination is a top-20 busiest airport |
| `IS_EVENING` | binary | 1 if departure is at 17h or later — evening flights accumulate delays from earlier flights |
| `IS_EARLY` | binary | 1 if departure is at 7h or earlier — early morning flights historically have the fewest delays |
| `IS_LONG_HAUL` | binary | 1 if distance > 1500 miles |
| `IS_HOLIDAY` | binary | 1 if the date falls on or near a major US holiday (New Year, July 4th, Thanksgiving, Christmas, Memorial Day, Labor Day) |

### Historical Pattern Features (target-encoded, leakage-safe)

These features represent the historical delay rate for a given category, computed using Bayesian smoothing on the training set only. They answer: "historically, how often do flights in this category arrive late?"

| Feature | Type | Description |
|---------|------|-------------|
| `ORIGIN_TE` | float | Delay rate of the origin airport |
| `DEST_TE` | float | Delay rate of the destination airport |
| `AIRLINE_TE` | float | Delay rate of the airline |
| `ROUTE_TE` | float | Delay rate of this specific origin→destination route |

### Interaction Features (target-encoded, leakage-safe)

These are the most powerful features. Instead of looking at airline and hour separately, interaction features capture combined patterns. For example, Spirit Airlines at 7PM in July has a very different delay profile than Delta at 7AM in April.

| Feature | Type | Description |
|---------|------|-------------|
| `AIRLINE_HOUR_TE` | float | Delay rate for this airline at this departure hour — captures airline-specific rush hour performance |
| `ORIGIN_HOUR_TE` | float | Delay rate for this airport at this hour — captures airport congestion patterns by time of day |
| `AIRLINE_MONTH_TE` | float | Delay rate for this airline in this month — captures seasonal airline performance |
| `ROUTE_DOW_TE` | float | Delay rate for this route on this day of week — captures route-specific weekly patterns |
| `DEST_HOUR_TE` | float | Delay rate for the destination airport at the arrival hour — captures arrival congestion |

---

## Target Encoding — How It Works

Standard one-hot encoding creates a column for each category (e.g., 350+ airports = 350 columns). This is sparse and memory-intensive. Target encoding replaces each category with a single number: its historical delay rate.

**Bayesian Smoothing** prevents overfitting on rare categories. A route with only 3 flights and 2 delays would naively get a 67% delay rate, which is unreliable. Smoothing pulls rare categories toward the global average:

```
smoothed_rate = (count / (count + smoothing)) * category_mean + (smoothing / (count + smoothing)) * global_mean
```

With `smoothing=100`, a category needs about 100 flights before its own rate dominates over the global average.

**Leakage Prevention:** The encoding is fitted on training labels only. The test set is transformed using the mapping learned from training data. If a test category was never seen in training, it falls back to the global mean.

---

## Data Preprocessing Pipeline

### Step 1 — Clean raw data
- Fill `DELAY_DUE_*` columns with 0 (NaN means no delay, not missing data)
- Drop cancelled and diverted flights (no arrival = no prediction possible)
- Drop rows with missing core scheduling columns
- Remove redundant columns (`AIRLINE_DOT`, `AIRLINE`, `CANCELLATION_CODE`)

### Step 2 — Engineer features
- Extract temporal features from `FL_DATE` and `CRS_DEP_TIME`
- Create binary risk flags (evening, early, holiday, busy airport, long haul)
- Build route identifier (`ORIGIN_DEST`)
- Define binary target: `IS_DELAYED = (ARR_DELAY > 15)`

### Step 3 — Balance the dataset
- Original distribution: 82% on time, 18% delayed (4.7:1 ratio)
- Undersample majority class to 65/35 ratio using random undersampling
- This preserves all delayed flights while reducing the on-time majority

### Step 4 — Train/test split (BEFORE encoding)
- 80/20 stratified split
- Splitting before encoding prevents data leakage

### Step 5 — Target encode categoricals
- Fit Bayesian-smoothed target encoding on train labels only
- Apply to both train and test
- Encode: ORIGIN, DEST, AIRLINE, ROUTE, plus 5 interaction features

### Step 6 — Scale numeric features
- StandardScaler on `CRS_ELAPSED_TIME` and `DISTANCE`
- Fitted on train, transformed on both

---

## Models

### HistGradientBoostingClassifier (Primary)
- `max_iter=800` with early stopping (patience=50)
- `max_depth=7`, `learning_rate=0.03`
- `l2_regularization=0.5` to prevent overfitting
- `min_samples_leaf=30`
- Balanced sample weights

### RandomForestClassifier (Secondary)
- `n_estimators=400`, `max_depth=18`
- `min_samples_leaf=10`, `min_samples_split=20`
- `class_weight='balanced'`

---

## Evaluation Metrics

| Metric | What it tells us |
|--------|-----------------|
| **Accuracy** | Overall correct predictions |
| **Precision (Delayed)** | Of flights predicted as delayed, how many actually were? |
| **Recall (Delayed)** | Of flights that were actually delayed, how many did we catch? |
| **F1 (Delayed)** | Harmonic mean of precision and recall for the delayed class |
| **ROC-AUC** | Model's ability to distinguish delayed from on-time across all thresholds |

---

## API Deployment

The trained model is served via a FastAPI backend with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Single flight prediction |
| `/predict/batch` | POST | Batch prediction for multiple flights |
| `/docs` | GET | Interactive Swagger documentation |

### Running the API

```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` to test predictions.

---

## Project Structure

```
project/
├── flights_sample_3m.csv          # Raw dataset (3M rows)
├── FlightDelay_PreFlight_Final.ipynb  # Complete notebook
├── best_model.pkl                 # Trained HistGradientBoosting model
├── scaler.pkl                     # Fitted StandardScaler
├── feature_names.pkl              # Feature column names (25 features)
├── mlruns/                        # MLflow experiment tracking
├── api/
│   ├── app.py                     # FastAPI application
│   ├── index.html                 # Frontend dashboard
│   ├── best_model.pkl             # Model copy for API
│   ├── scaler.pkl                 # Scaler copy for API
│   └── feature_names.pkl          # Features copy for API
└── README.md
```

---

## Key Takeaways

1. **Feature engineering matters more than model complexity.** The jump from 64% to 75%+ came from interaction features, not from changing the algorithm.

2. **Data leakage is silent and dangerous.** Computing target encodings before the train/test split inflated results without improving real-world performance.

3. **Pre-flight prediction is inherently limited.** Without knowing the actual departure delay, the ceiling for accuracy is around 75-80%. This is realistic — weather, mechanical issues, and cascading delays are unpredictable.

4. **Class imbalance requires careful handling.** Only 18% of flights are delayed. Without undersampling and balanced class weights, the model would just predict "on time" for everything and score 82% accuracy while being completely useless for the delayed class.

---

## Tools & Libraries

- **Python 3.13** — Runtime
- **pandas / numpy** — Data manipulation
- **scikit-learn** — Models, preprocessing, evaluation
- **matplotlib / seaborn** — Visualization
- **MLflow** — Experiment tracking
- **FastAPI / Uvicorn** — API deployment
- **joblib** — Model serialization
>>>>>>> 14f4fb8 (first commit)
