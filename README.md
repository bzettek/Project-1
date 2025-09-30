# CFB Attendance Predictor

A full-stack project that forecasts college football game attendance from historical data, exposes the model through a Flask API, and presents a responsive single-page interface for interactive predictions. This repository also includes training utilities and a Jupyter walkthrough so you can understand, retrain, and validate the model end-to-end.

----

## Table of Contents
1. [System Overview](#system-overview)
2. [Data & Feature Engineering](#data--feature-engineering)
3. [Model Architecture](#model-architecture)
4. [Inference Adjustments](#inference-adjustments)
5. [API Surface](#api-surface)
6. [Web UI](#web-ui)
7. [Running Locally](#running-locally)
8. [Retraining the Model](#retraining-the-model)
9. [Model Walkthrough Notebook](#model-walkthrough-notebook)
10. [Project Structure](#project-structure)
11. [Deployment Notes](#deployment-notes)

---

## System Overview
```
                    ┌─────────────────────┐
                    │  merged_attendance  │
                    │     _dataset.csv    │
                    └──────────┬──────────┘
                               │
                        train_new_model.py
                               │
                               ▼
┌────────────────────────┐   new_regressor_model.joblib   ┌───────────────────────┐
│  Flask API (app.py)    │◄───────────────────────────────┤  model_utils.py       │
│  /api/team /api/predict│                                │  augment_features()   │
└──────────┬─────────────┘                                └──────────┬────────────┘
           │                                                       │
           ▼                                                       ▼
 ┌────────────────────────────┐                         notebooks/model_walkthrough.ipynb
 │  Frontend (templates/ +    │
 │  static/) – responsive UI  │
 └────────────────────────────┘
```
The workflow looks like this:
1. **Data** is aggregated into `merged_attendance_dataset.csv` (already provided) containing per-game stats (stadium capacity, fill rate, win/loss record, weather, etc.).
2. **Training** (`train_new_model.py`) builds a RandomForest pipeline with engineered features and writes both the serialized model (`new_regressor_model.joblib`) and evaluation metrics (`model_metrics.json`).
3. **Serving** (`app.py`) loads the Joblib pipeline, applies domain-specific adjustments, and exposes `/api/team` and `/api/predict` endpoints used by both the web UI and external clients.
4. **Frontend** (`templates/index.html` + `static/styles.css`) provides a mobile-friendly interface that guides users through data entry, performs client-side validation, and requests predictions from the API.
5. **Exploration** (`notebooks/model_walkthrough.ipynb`) documents the entire process with live code, plots, and commentary.

---

## Data & Feature Engineering
The merged dataset combines multiple sources. Key raw columns:
- **`Stadium Capacity` / `AutoCapacity`** – official or scraped seating capacity.
- **`Fill Rate` / `AutoFillRate`** – historical attendance as a fraction of capacity.
- **`Current Wins`, `Current Losses`** – team record entering the game.
- **`PRCP`** – precipitation (inches) recorded for game day.

During preprocessing we create:
- **`cap_unified`**: choose between manual and automated capacity fields.
- **`fill_unified`**: choose between manual and automated fill rate, converting percentages to decimals if needed.
- **Interaction Features** (computed inside `model_utils.augment_features`):
  - Win percentage (`wins / (wins + losses)`), safe for zero-loss teams.
  - Expected attendance (`capacity * fill`).
  - Slack capacity (`capacity - expected`) capturing unused seats.
  - Net record (`wins - losses`), rain pressure (`prcp * capacity`), log-capacity, etc.

These engineered features give the tree-based model richer, non-linear signals beyond the raw five inputs accepted by the UI/API.

---

## Model Architecture
We favour a **RandomForestRegressor** because it is:
- Robust to outliers and non-linear patterns.
- Compatible across scikit-learn versions when serialized.
- Interpretable via feature importance scores.

The training pipeline:
1. `SimpleImputer(strategy="median")` – replaces any missing numeric input.
2. `FunctionTransformer(augment_features)` – expands the five base features into the engineered set described above.
3. `RandomForestRegressor(n_estimators=400, min_samples_leaf=20)` – ensemble of decision trees, seeded for reproducibility.

Evaluation metrics written to `model_metrics.json` include:
- **RMSE** (~718 seats on holdout set).
- **MAE** (~341 seats).
- **R²** (> 0.999), indicating excellent explanatory power on historical data.

Refer to `train_new_model.py` for implementation details.

---

## Inference Adjustments
While the RandomForest produces strong predictions, the API applies a few deterministic business rules before returning a value:
1. **Fill-rate clamp** – incoming fill values are clamped to `[0, 1]`.
2. **Rain penalty** – if precipitation is reported, we dampen the model output by up to 12% (for heavy rain) to mirror observed drops on wet days.
3. **Record boost/punishment** – positive net records boost attendance (up to +30%), losing records reduce it (up to −45%). This emphasises momentum without retraining the model for each season.
4. **Capacity ceiling** – predictions are capped at the provided stadium capacity to avoid implausible sell-through.

These post-processing steps happen in `app.py` inside `/api/predict` and keep the output aligned with domain expectations.

---

## API Surface
`app.py` exposes two primary endpoints plus a health check:

| Route            | Method | Description                                                                 | Example |
|------------------|--------|-----------------------------------------------------------------------------|---------|
| `/api/teams`     | GET    | Returns list of all known team names/aliases for the auto-complete UI.      | `curl http://localhost:8080/api/teams` |
| `/api/team`      | GET    | Given a `name` query param, performs fuzzy matching and returns stadium capacity/fill defaults if known. | `curl "http://localhost:8080/api/team?name=Notre%20Dame"` |
| `/api/predict`   | POST   | Accepts JSON `{"features": [capacity, fill, wins, losses, prcp], "extras": {...}}` and responds with `{"prediction": value}`. | See `prediction.py` for usage |
| `/health`        | GET    | Simple health check returning `ok`.                                        |         |

The web UI consumes these endpoints, but they are equally usable from scripts or other services.

---

## Web UI
- Located under `templates/` (HTML) and `static/` (CSS + assets).
- Fully responsive: on desktop, form fields are arranged in grids; on mobile, they stack vertically.
- The home team field offers live suggestions and auto-fills capacity/fill when possible.
- Tooltips provide inline definitions (e.g., what "fill rate" means) with tap-friendly behaviour on touch devices.
- Client-side validation ensures numerical inputs are present and within allowed ranges before hitting the API.
- Prediction results are displayed inline with spinner/disabled states for user feedback.

Screenshots and design rationale live inside the markup via comments and class names.

---

## Running Locally
1. **Create/activate a virtual environment** (optional but recommended).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the Flask dev server:**
   ```bash
   flask --app app.py --debug run
   ```
   The app listens on [http://localhost:5000](http://localhost:5000) by default.
4. **Visit the UI** at `/` and test predictions. Use the Notebook (below) for deeper verification.

> **Tip:** When running outside the repo root (e.g., inside `notebooks/`), make sure to set `PYTHONPATH` or adjust `sys.path` so `model_utils` is importable.

---

## Retraining the Model
To retrain with updated data or hyperparameters:
1. Replace or edit `merged_attendance_dataset.csv`.
2. Optionally tweak `train_new_model.py` (feature engineering, hyperparameters, etc.).
3. Run:
   ```bash
   python train_new_model.py
   ```
4. Commit/deploy the regenerated `new_regressor_model.joblib` and `model_metrics.json` alongside the code.
5. Restart the Flask/Gunicorn process to pick up the new artifact.

> **Warning:** Scikit-learn versions must match between training and serving. The current pipeline relies only on stable components (`SimpleImputer`, `FunctionTransformer`, `RandomForestRegressor`) to remain cross-version friendly.

---

## Model Walkthrough Notebook
`notebooks/model_walkthrough.ipynb` mirrors the steps above with extensive explanations, plots, and metrics. Highlights:
- Resolves dataset/model paths automatically so the notebook runs from any directory.
- Adds the repository root to `sys.path` so `model_utils` is importable.
- Documents every concept: feature engineering, evaluation metrics, residual analysis, feature importances, and next steps.
- Designed for newcomers who have a CS background but are new to ML pipelines.

Open the notebook in JupyterLab/VS Code, execute each cell sequentially, and use it as both documentation and a regression test harness.

---

## Project Structure
```
.
├── app.py                     # Flask application + API endpoints
├── prediction.py              # CLI-style helper for quick predictions
├── train_new_model.py         # Training script (RandomForest pipeline)
├── model_utils.py             # Shared feature engineering utility
├── model_metrics.json         # Latest evaluation metrics (generated)
├── new_regressor_model.joblib # Serialized model pipeline (generated)
├── notebooks/
│   └── model_walkthrough.ipynb# Detailed documentation + analysis notebook
├── templates/
│   ├── index.html             # Main UI
│   ├── how.html               # Project background page
│   └── why.html               # Methodology / motivation page
├── static/
│   ├── styles.css             # Global styles (desktop + mobile)
│   └── cfbLogo.png            # Logo asset
├── merged_attendance_dataset.csv # Training/evaluation dataset
├── requirements.txt
└── ...                        # Additional scripts/data
```

---

## Deployment Notes
- The production environment currently runs scikit-learn 1.7.x. The model artifact is trained with a compatible version and avoids custom Cython classes, so unpickling succeeds across minor releases.
- When deploying to Gunicorn/Heroku:
  - Ensure `model_utils.py` ships with the code so `augment_features` resolves during unpickling.
  - Include `requirements.txt` updates (Matplotlib/Seaborn are optional for runtime but required for the notebook).
  - After pushing new artifacts, restart workers to load the updated model.
- Logging: `/api/predict` appends requests to `data/predictions_log.csv` (created on demand). Tail this to monitor usage or build monitoring dashboards.

---

**Need more detail?** Explore `train_new_model.py` and the notebook together—the inline comments, docstrings, and plots provide a gentle but thorough introduction to each component of the system.
