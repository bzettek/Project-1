# app.py — CFB Attendance Predictor (Page 1)
# Endpoints:
#   GET  /                 -> predictor UI
#   GET  /api/teams        -> list of team names for the datalist
#   GET  /api/team?name=.. -> {team, capacity?, fill?} with fuzzy match
#   POST /api/predict      -> {features:[cap,fill,wins,losses,prcp], extras:{...}} -> {prediction}
#
# Notes:
# - Team names, capacities, and fill rates load exclusively from update_tNames_cap_fill.csv (override via TEAM_SOURCE_CSV env).
# - Fill is clamped to [0,1].
# - Logs predictions + context to data/predictions_log.csv.

from flask import Flask, render_template, jsonify, request
import numpy as np
import os, csv, time, re
from joblib import load
import pandas as pd

# ---------------------- Config ----------------------
MODEL_PATH    = os.getenv("MODEL_PATH",    "new_regressor_model.joblib")  # use the compressed joblib
TEAM_SOURCE_CSV = os.getenv("TEAM_SOURCE_CSV", "update_tNames_cap_fill.csv")
EXPECTED      = 5  # [capacity, fill (0–1), wins, losses, prcp]

app = Flask(__name__, template_folder="templates", static_folder="static")
model = load(MODEL_PATH)

# ---------------------- Catalog / Lookup ----------------------
LOOKUP = {}          # normalized team name -> {team, capacity, fill}
TEAM_NAMES_ALL = []  # for UI datalist

def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def _split_tokens(name: str) -> list[str]:
    """Lightweight tokenizer used only for matching (no alias generation)."""
    return re.findall(r"[a-z0-9]+", (name or "").lower())

def _build_from_team_source(path: str) -> bool:
    """Populate lookup exclusively from the curated CSV with Team / Capacity / Fill Rate."""
    if not os.path.exists(path):
        return False

    df = pd.read_csv(path)
    required = {"Team", "Capacity", "Fill Rate"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Team source CSV is missing required columns: {missing}")

    # Normalise numeric inputs first (strip commas, force numeric)
    df["Capacity"] = (
        df["Capacity"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace({"": None})
    )
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df["Fill Rate"] = pd.to_numeric(df["Fill Rate"], errors="coerce")

    # Reset global containers
    LOOKUP.clear()
    TEAM_NAMES_ALL.clear()

    # Deduplicate by the provided team display name (last entry wins)
    for team, group in df.groupby("Team", sort=True):
        team_name = str(team).strip()
        if not team_name:
            continue

        key = _normalize_name(team_name)
        if not key:
            continue

        cap_series = group["Capacity"].dropna()
        capacity = int(round(float(cap_series.iloc[-1]))) if not cap_series.empty else None

        fill_series = group["Fill Rate"].dropna()
        fill = float(fill_series.iloc[-1]) if not fill_series.empty else None
        if fill is not None:
            if fill > 1.5:
                fill /= 100.0
            fill = max(0.0, min(fill, 1.0))

        LOOKUP[key] = {
            "team": team_name,
            "capacity": capacity,
            "fill": fill,
            "tokens": _split_tokens(team_name),
        }
        TEAM_NAMES_ALL.append(team_name)

    TEAM_NAMES_ALL[:] = sorted(set(TEAM_NAMES_ALL))
    return True

# Build data exclusively from the curated team list CSV
if not _build_from_team_source(TEAM_SOURCE_CSV):
    raise RuntimeError(f"Unable to load team data from {TEAM_SOURCE_CSV}")

# Optional manual overrides example:
# OVERRIDES = {"notre dame": {"capacity": 80795}}
# for k, v in OVERRIDES.items():
#     if k in LOOKUP:
#         LOOKUP[k].update(v)

# ---------------------- Helpers ----------------------
def _best_key_for(query: str):
    """Return the best-matching team key based on straightforward string matching."""
    q = (query or "").strip()
    if not q:
        return None

    q_norm = _normalize_name(q)

    # direct normalized key hit
    if q_norm in LOOKUP:
        return q_norm

    # exact name match ignoring case
    for key, data in LOOKUP.items():
        if data["team"].lower() == q_norm:
            return key

    # prefix matches
    prefix_matches = [
        key for key, data in LOOKUP.items()
        if data["team"].lower().startswith(q_norm)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if prefix_matches:
        return min(prefix_matches, key=lambda k: len(LOOKUP[k]["team"]))

    # substring matches
    substring_matches = [
        key for key, data in LOOKUP.items()
        if q_norm in data["team"].lower()
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if substring_matches:
        return min(substring_matches, key=lambda k: len(LOOKUP[k]["team"]))

    # token overlap scoring (helps queries like "university of notre dame")
    query_tokens = _split_tokens(q)
    if not query_tokens:
        return None

    best_key = None
    best_score = 0
    best_length = None

    for key, data in LOOKUP.items():
        team_tokens = data.get("tokens") or _split_tokens(data["team"])
        score = 0
        for token in query_tokens:
            if token in team_tokens:
                score += 2
            elif any(t.startswith(token) for t in team_tokens):
                score += 1
        if score > best_score:
            best_key = key
            best_score = score
            best_length = len(data["team"])
        elif score == best_score and score > 0:
            name_len = len(data["team"])
            if best_length is None or name_len < best_length:
                best_key = key
                best_length = name_len

    return best_key if best_score > 0 else None

# ---------------------- Routes ----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/how")
def how_page():
    return render_template("how.html")

@app.route("/why")
def why_page():
    return render_template("why.html")

@app.route("/api/teams")
def api_teams():
    return jsonify(TEAM_NAMES_ALL)

@app.route("/api/team")
def api_team():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({}), 400
    tkey = _best_key_for(name)
    if tkey and tkey in LOOKUP:
        record = LOOKUP[tkey]
        info = {"team": record["team"]}
        cap = record.get("capacity")
        fill = record.get("fill")
        if cap is not None:
            info["capacity"] = int(cap)
        if fill is not None:
            info["fill"] = max(0.0, min(float(fill), 1.0))
        return jsonify(info)
    return jsonify({"team": name})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True) or {}
        vals = data.get("features", [])
        extras = data.get("extras", {}) or {}

        if not isinstance(vals, list):
            return jsonify({"error": "features must be a list"}), 400
        if len(vals) != EXPECTED:
            return jsonify({"error": f"Expected {EXPECTED} numbers [capacity, fill, wins, losses, prcp]"}), 400

        try:
            vals = [float(x) for x in vals]
        except Exception:
            return jsonify({"error": "features must be numeric"}), 400

        # clamp fill
        vals[1] = max(0.0, min(vals[1], 1.0))

        x = np.array(vals, dtype=float).reshape(1, -1)
        raw_pred = float(model.predict(x)[0])

        capacity_input = max(0.0, vals[0])
        prcp_input = max(0.0, vals[4])

        # Apply light precipitation penalty to respect expected downturn on wet days
        if prcp_input > 0:
            damp_factor = max(0.0, 1.0 - min(prcp_input, 1.5) * 0.08)
            raw_pred *= damp_factor

        # Reward positive records (more wins vs losses) a bit more aggressively
        net_record = vals[2] - vals[3]
        if net_record > 0:
            raw_pred *= (1.0 + min(net_record * 0.025, 0.30))
        elif net_record < 0:
            raw_pred *= max(0.0, 1.0 - min(abs(net_record) * 0.04, 0.45))

        pred = int(round(max(0.0, min(raw_pred, capacity_input))))

        # Log (best-effort)
        try:
            os.makedirs("data", exist_ok=True)
            log_path = os.path.join("data", "predictions_log.csv")
            write_header = not os.path.exists(log_path)
            with open(log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts","capacity","fill","wins","losses","prcp","prediction","user_ip",
                    "homeTeam","rankedStatus","rivalry","kickoffWindow"
                ])
                if write_header: w.writeheader()
                w.writerow({
                    "ts": int(time.time()),
                    "capacity": vals[0], "fill": vals[1], "wins": vals[2], "losses": vals[3], "prcp": vals[4],
                    "prediction": pred,
                    "user_ip": request.headers.get("CF-Connecting-IP") or request.remote_addr,
                    "homeTeam": extras.get("homeTeam"),
                    "rankedStatus": extras.get("rankedStatus"),
                    "rivalry": extras.get("rivalry"),
                    "kickoffWindow": extras.get("kickoffWindow")
                })
        except Exception:
            pass

        return jsonify({"prediction": pred})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400

@app.route("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    # Use PORT env (App Platform injects one) or default 8080 locally
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)
