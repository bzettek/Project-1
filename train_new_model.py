import pandas as pd, numpy as np, json, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

CSV_PATH = "merged_attendance_dataset.csv"
MODEL_PATH = "new_regressor_model.pkl"
METRICS_PATH = "model_metrics.json"

df = pd.read_csv(CSV_PATH)

# unify capacity/fill
df["cap_unified"]  = df["Stadium Capacity"].where(df["Stadium Capacity"].notna(), df["AutoCapacity"])
df["fill_unified"] = df["Fill Rate"].where(df["Fill Rate"].notna(), df["AutoFillRate"])
if df["fill_unified"].dropna().max() > 1.5:
    df["fill_unified"] = df["fill_unified"] / 100.0

features = ["cap_unified", "fill_unified", "Current Wins", "Current Losses", "PRCP"]
target   = "Attendance"
dfm = df[features + [target]].dropna(subset=[target])

X, y = dfm[features], dfm[target]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
])
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)

metrics = {
    "rmse": float(mean_squared_error(yte, pred, squared=False)),
    "mae":  float(mean_absolute_error(yte, pred)),
    "r2":   float(r2_score(yte, pred)),
    "features_order": features
}
dump(pipe, MODEL_PATH)
with open(METRICS_PATH, "w") as f: json.dump(metrics, f, indent=2)
print("Saved:", os.path.abspath(MODEL_PATH))
print("Metrics:", metrics)