# app.py — CFB Attendance Predictor (Page 1)
# Endpoints:
#   GET  /                 -> predictor UI
#   GET  /api/teams        -> list of ALL team names + aliases (for datalist)
#   GET  /api/team?name=.. -> {team, capacity?, fill?} with fuzzy match
#   POST /api/predict      -> {features:[cap,fill,wins,losses,prcp], extras:{...}} -> {prediction}
#
# Features:
# - Uses MAX capacity across rows per team (handles stadium upgrades, e.g., Notre Dame 80,795).
# - Fill is clamped to [0,1].
# - Builds aliases (e.g., "Michigan", "Oregon") and adds forced must-have aliases (e.g., "Ohio State").
# - Fuzzy lookup so "Ohio State" / "Ohio St." / "The Ohio State University" all resolve.
# - Logs predictions + context to data/predictions_log.csv.

from flask import Flask, render_template, jsonify, request
import numpy as np
import os, json, csv, time, re
from joblib import load
import pandas as pd

MODEL_PATH   = os.getenv("MODEL_PATH",   "new_regressor_model.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "model_metrics.json")
MERGED_CSV   = os.getenv("MERGED_CSV",   "merged_attendance_dataset.csv")

app = Flask(__name__, template_folder="templates", static_folder="static")
model = load(MODEL_PATH)

# The model expects EXACTLY 5 features, in this order:
# [capacity, fill (0–1), wins, losses, prcp]
EXPECTED = 5

# ---------------------- Utilities ----------------------

STOPWORDS = {"university", "of", "the", "and", "&", "at"}
SECOND_WORD_KEEP = {
    "state", "tech", "a&m", "aandm", "southern", "western", "eastern",
    "northern", "central", "poly", "city", "international"
}

def _norm_team(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\bst\.\b", "state", s)
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _tokens(s: str):
    return [t for t in _norm_team(s).split() if t and t not in STOPWORDS]

def _alias_variants(display: str):
    """Generate useful short names like 'Michigan', 'Oregon', 'Texas State'."""
    al = set()
    disp = display.strip()
    if not disp: return al

    toks = _tokens(disp)
    if not toks: return al

    # 1) Single-token alias (first token) e.g., "Michigan", "Oregon"
    al.add(toks[0].title())

    # 2) Two-token prefix if second token is a keeper (e.g., "Texas State", "Georgia Tech")
    if len(toks) >= 2 and toks[1] in SECOND_WORD_KEEP:
        al.add((toks[0] + " " + toks[1]).title())

    # 3) "University of X" -> X
    m = re.search(r"university of ([a-z\s]+)$", _norm_team(disp))
    if m:
        x = m.group(1).strip().title()
        if x: al.add(x)

    # 4) Remove trailing parentheses "X (Football)" -> "X"
    al.add(re.sub(r"\s*\([^)]*\)\s*$", "", disp).strip())

    # 5) Heuristic mascot strip: if last word ends with 's', keep base like "Michigan" or "Texas State"
    if len(toks) >= 2 and disp.split()[-1].endswith("s"):
        base = [toks[0]]
        if len(toks) >= 2 and toks[1] in SECOND_WORD_KEEP:
            base.append(toks[1])
        al.add(" ".join(w.title() for w in base).strip())

    return {x for x in al if x and len(x) >= 2}

# ---------------------- Build Lookup, Aliases, Team List ----------------------

LOOKUP = {}           # team_key -> { team, capacity, fill }
DISPLAY_NAME = {}     # team_key -> chosen display name
TEAM_ALIASES = {}     # alias -> team_key (for quick reverse lookup)
TEAM_NAMES_ALL = []   # list[str] for datalist (display names + aliases)

if os.path.exists(MERGED_CSV):
    df = pd.read_csv(MERGED_CSV)

    # Identify columns that may contain team names (inclusive search)
    name_like = [c for c in df.columns if any(k in c.lower() for k in ["team", "school", "program", "home team", "away team"])]
    if "Team" not in name_like and "Team" in df.columns:
        name_like.append("Team")

    # Choose display per normalized key by frequency (tie-break by length)
    display_counts = {}
    for col in name_like:
        if col not in df.columns: continue
        for val in df[col].dropna().astype(str):
            key = _norm_team(val)
            if not key: continue
            display_counts.setdefault(key, {})
            display_counts[key][val] = display_counts[key].get(val, 0) + 1

    for key, variants in display_counts.items():
        disp = sorted(variants.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[0][0]
        DISPLAY_NAME[key] = disp

    # Candidate capacity/fill per row
    cap_col  = "Stadium Capacity" if "Stadium Capacity" in df.columns else None
    fill_col = "Fill Rate"        if "Fill Rate"        in df.columns else None

    def row_capacity(r):
        if cap_col and pd.notna(r.get(cap_col)): return float(r[cap_col])
        if pd.notna(r.get("AutoCapacity")):      return float(r["AutoCapacity"])
        if pd.notna(r.get("2022AvgAttendance")) and pd.notna(r.get("2022CapacityPct")):
            try:
                pct = float(str(r["2022CapacityPct"]).rstrip("%")) / 100.0
                if pct > 0: return float(r["2022AvgAttendance"]) / pct
            except Exception:
                pass
        return None

    def row_fill(r):
        v = None
        if fill_col and pd.notna(r.get(fill_col)): v = float(r[fill_col])
        elif pd.notna(r.get("AutoFillRate")):      v = float(r["AutoFillRate"])
        elif pd.notna(r.get("FiveYearCapacityPct")):
            try: v = float(str(r["FiveYearCapacityPct"]).rstrip("%")) / 100.0
            except Exception: v = None
        if v is not None and v > 1.5: v = v / 100.0
        if v is not None: v = max(0.0, min(v, 1.0))
        return v

    # unify team key per row
    def pick_team_val(row):
        for col in (["Team"] + [c for c in name_like if c != "Team"]):
            if col in row and pd.notna(row[col]):
                s = str(row[col]).strip()
                if s: return s
        return ""

    df["_team_display_raw"] = df.apply(pick_team_val, axis=1)
    df["_team_key"]         = df["_team_display_raw"].map(_norm_team)
    df["_cap_candidate"]    = df.apply(row_capacity, axis=1)
    df["_fill_candidate"]   = df.apply(row_fill, axis=1)

    # aggregate per team
    agg = df.groupby("_team_key").agg(
        cap_max  = ("_cap_candidate", "max"),
        fill_med = ("_fill_candidate", "median")
    ).reset_index(drop=False)

    # build LOOKUP
    for _, r in agg.iterrows():
        key = r["_team_key"]
        if not key: continue
        disp = DISPLAY_NAME.get(key) or df.loc[df["_team_key"] == key, "_team_display_raw"].dropna().astype(str).iloc[0]
        cap = r["cap_max"]; fill = r["fill_med"]
        if pd.notna(cap):
            LOOKUP[key] = {
                "team": disp,
                "capacity": int(round(float(cap))),               # MAX capacity
                "fill": float(fill) if pd.notna(fill) else None   # MEDIAN fill
            }

    # Build aliases -> team_key
    for key, disp in DISPLAY_NAME.items():
        TEAM_ALIASES[disp] = key  # map the display itself
        for a in _alias_variants(disp):
            TEAM_ALIASES.setdefault(a, key)

    # ---- Force-add common short aliases for major programs ----
    MUST_HAVE_ALIASES = {
        "Ohio State": ["Ohio St.", "The Ohio State University", "Ohio State University"],
        "Oregon": ["Oregon Ducks", "University of Oregon"],
        "Michigan": ["Michigan Wolverines", "University of Michigan"],
        "Texas A&M": ["Texas A and M", "Texas A&M University"],
        "Ole Miss": ["Mississippi", "University of Mississippi"],
        "USC": ["Southern California", "University of Southern California"],
        "UCLA": ["University of California Los Angeles", "California-Los Angeles"],
        "Pitt": ["Pittsburgh", "University of Pittsburgh"],
        "NC State": ["North Carolina State"],
        "UTSA": ["Texas San Antonio", "Texas at San Antonio"],
        "BYU": ["Brigham Young", "Brigham Young University"],
        "SMU": ["Southern Methodist", "Southern Methodist University"],
        "Miami": ["Miami (FL)", "Miami (Florida)"],
        "Miami (OH)": ["Miami (Ohio)"],
        "Arizona State": ["Arizona St."],
        "Colorado State": ["Colorado St."],
        "Fresno State": ["Fresno St."],
        "Kansas State": ["Kansas St."],
        "Oklahoma State": ["Oklahoma St."],
        "San Diego State": ["San Diego St."],
        "San Jose State": ["San Jose St."],
        "Boise State": ["Boise St."],
        "Washington State": ["Washington St."],
        "Utah State": ["Utah St."],
        "Ball State": ["Ball St."],
        "Bowling Green": ["Bowling Green State"],
    }

    def _best_key_for_alias(name: str):
        q = name.strip()
        if not q: return None
        k = _norm_team(q)
        if k in DISPLAY_NAME:  # exact normalized display
            return k
        # token-overlap against DISPLAY_NAME
        qtok = set(_tokens(q))
        best, best_score = None, -1
        for tkey, disp in DISPLAY_NAME.items():
            dtok = set(_tokens(disp))
            score = len(qtok & dtok)
            if score > best_score:
                best, best_score = tkey, score
        return best

    for short_name, synonyms in MUST_HAVE_ALIASES.items():
        tkey = _best_key_for_alias(short_name)
        if tkey:
            TEAM_ALIASES[short_name] = tkey
            for s in synonyms:
                TEAM_ALIASES.setdefault(s, tkey)

    # Public list = union of display names and all aliases (incl. forced)
    TEAM_NAMES_ALL = sorted(set(
        list(DISPLAY_NAME.values()) +
        list(TEAM_ALIASES.keys())
    ))

# Optional manual overrides for known upgrades
# OVERRIDES = { "notre dame": {"capacity": 80795} }
# for k, v in OVERRIDES.items():
#     if k in LOOKUP: LOOKUP[k].update(v)

# ---------------------- Routes ----------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/teams")
def api_teams():
    return jsonify(TEAM_NAMES_ALL)

def _best_key_for(query: str):
    """Exact key -> alias hit -> normalized alias -> token-overlap fuzzy best."""
    q = query.strip()
    if not q: return None
    k = _norm_team(q)
    # direct hit by normalized key
    if k in LOOKUP or k in DISPLAY_NAME:
        return k if k in DISPLAY_NAME else None
    # alias hit (case-sensitive display or short)
    if q in TEAM_ALIASES:
        return TEAM_ALIASES[q]
    # normalized alias map
    norm_alias_map = getattr(_best_key_for, "_norm_alias_map", None)
    if norm_alias_map is None:
        norm_alias_map = {}
        for alias, tkey in TEAM_ALIASES.items():
            norm_alias_map.setdefault(_norm_team(alias), tkey)
        _best_key_for._norm_alias_map = norm_alias_map
    if k in norm_alias_map:
        return norm_alias_map[k]
    # token overlap against DISPLAY_NAME
    qtok = set(_tokens(q))
    best, best_score = None, 0
    for tkey, disp in DISPLAY_NAME.items():
        dtok = set(_tokens(disp))
        score = len(qtok & dtok)
        if score > best_score:
            best, best_score = tkey, score
    return best

@app.route("/api/team")
def api_team():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({}), 400
    tkey = _best_key_for(name)
    if tkey and tkey in DISPLAY_NAME:
        info = {"team": DISPLAY_NAME[tkey]}
        if tkey in LOOKUP:
            cap = LOOKUP[tkey].get("capacity")
            fill = LOOKUP[tkey].get("fill")
            if cap is not None:  info["capacity"] = int(cap)
            if fill is not None: info["fill"] = max(0.0, min(float(fill), 1.0))
        return jsonify(info)
    # fallback: echo back
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
        pred = round(float(model.predict(x)[0]))

        # Logging (non-blocking)
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)

@app.route("/health")
def health():
    return "ok", 200