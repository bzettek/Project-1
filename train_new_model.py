"""Train refined attendance regressor with monotonic constraints."""

import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

CSV_PATH = "merged_attendance_dataset.csv"
MODEL_PATH = "new_regressor_model.joblib"
METRICS_PATH = "model_metrics.json"
RANDOM_STATE = 42
FEATURES = ["cap_unified", "fill_unified", "Current Wins", "Current Losses", "PRCP"]
TARGET = "Attendance"


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["cap_unified"] = df["Stadium Capacity"].where(df["Stadium Capacity"].notna(), df["AutoCapacity"])
    df["fill_unified"] = df["Fill Rate"].where(df["Fill Rate"].notna(), df["AutoFillRate"])
    if df["fill_unified"].dropna().max() > 1.5:
        df["fill_unified"] = df["fill_unified"] / 100.0
    return df


def build_pipeline() -> Pipeline:
    regressor = HistGradientBoostingRegressor(
        max_depth=5,
        max_iter=450,
        learning_rate=0.08,
        min_samples_leaf=25,
        l2_regularization=0.4,
        monotonic_cst=(1, 1, 1, -1, -1),  # precip + losses decrease attendance
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", regressor),
    ])


def main() -> None:
    df = load_dataframe(CSV_PATH)
    data = df[FEATURES + [TARGET]].dropna(subset=[TARGET])

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    holdout_pred = pipeline.predict(X_test)

    rmse = float(mean_squared_error(y_test, holdout_pred, squared=False))
    mae = float(mean_absolute_error(y_test, holdout_pred))
    r2 = float(r2_score(y_test, holdout_pred))

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rmse = float(
        (-cross_val_score(
            pipeline,
            X,
            y,
            scoring="neg_root_mean_squared_error",
            cv=kfold,
            n_jobs=1,
        )).mean()
    )
    cv_mae = float(
        (-cross_val_score(
            pipeline,
            X,
            y,
            scoring="neg_mean_absolute_error",
            cv=kfold,
            n_jobs=1,
        )).mean()
    )
    cv_r2 = float(cross_val_score(pipeline, X, y, scoring="r2", cv=kfold, n_jobs=1).mean())

    metrics = {
        "holdout": {"rmse": rmse, "mae": mae, "r2": r2},
        "cv_mean": {"rmse": cv_rmse, "mae": cv_mae, "r2": cv_r2},
        "features_order": FEATURES,
        "model": "HistGradientBoostingRegressor",
        "random_state": RANDOM_STATE,
        "n_samples": int(len(data)),
    }

    dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to: {Path(MODEL_PATH).resolve()}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
