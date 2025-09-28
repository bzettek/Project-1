import os, json, numpy as np
from joblib import load

MODEL_PATH = os.getenv("MODEL_PATH", "new_regressor_model.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "model_metrics.json")
model = load(MODEL_PATH)

# The model expects 5 features, in this order:
# [cap_unified, fill_unified, Current Wins, Current Losses, PRCP]
EXPECTED = 5

def _ok(payload): return {"model": str(payload)}
def _err(msg, code=400): return {"model": f"ERROR: {msg}"}, code

def my_prediction(id):
    try:
        if not isinstance(id, (list, tuple)):
            return _err("Send comma-separated numbers in the path.")
        vals = [float(v) for v in id]
        if len(vals) != EXPECTED:
            return _err(f"Expected {EXPECTED} numbers but got {len(vals)}. "
                        f"Order: [cap_unified, fill_unified, wins, losses, prcp]")
        X = np.array(vals, dtype=float).reshape(1, -1)
        y = float(model.predict(X)[0])
        return _ok(round(y))
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 400)

def my_meteric():
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f: m = json.load(f)
            return _ok(json.dumps(m))
        return _ok("No metrics file found.")
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 400)