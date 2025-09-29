# app.py — CFB Attendance Predictor (Page 1)
# Endpoints:
#   GET  /                 -> predictor UI
#   GET  /api/teams        -> list of ALL team names + aliases (for datalist)
#   GET  /api/team?name=.. -> {team, capacity?, fill?} with fuzzy match
#   POST /api/predict      -> {features:[cap,fill,wins,losses,prcp], extras:{...}} -> {prediction}
#
# Notes:
# - Prefers a frozen catalog JSON (teams_catalog.json) so names are consistent, no CSV dependency.
# - If JSON is missing, it will build a lookup from MERGED_CSV (uses MAX capacity, MEDIAN fill).
# - Fill is clamped to [0,1].
# - Must-have aliases included (e.g., "Ohio St.", "The Ohio State University").
# - Logs predictions + context to data/predictions_log.csv.

from flask import Flask, render_template, jsonify, request
import numpy as np
import os, json, csv, time, re
from joblib import load
import pandas as pd

# ---------------------- Config ----------------------
MODEL_PATH    = os.getenv("MODEL_PATH",    "new_regressor_model.joblib")  # use the compressed joblib
MERGED_CSV    = os.getenv("MERGED_CSV",    "merged_attendance_dataset.csv")
CATALOG_PATH  = os.getenv("CATALOG_PATH",  "teams_catalog.json")
EXPECTED      = 5  # [capacity, fill (0–1), wins, losses, prcp]

app = Flask(__name__, template_folder="templates", static_folder="static")
model = load(MODEL_PATH)

# ---------------------- Text utils ----------------------
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
    disp = (display or "").strip()
    if not disp:
        return al
    toks = _tokens(disp)
    if not toks:
        return al
    al.add(toks[0].title())  # single-token short
    if len(toks) >= 2 and toks[1] in SECOND_WORD_KEEP:
        al.add((toks[0] + " " + toks[1]).title())  # two-token keeper
    m = re.search(r"university of ([a-z\s]+)$", _norm_team(disp))
    if m:
        x = m.group(1).strip().title()
        if x:
            al.add(x)
    al.add(re.sub(r"\s*\([^)]*\)\s*$", "", disp).strip())  # strip trailing parens
    if len(toks) >= 2 and disp.split()[-1].endswith("s"):
        base = [toks[0]]
        if toks[1] in SECOND_WORD_KEEP:
            base.append(toks[1])
        al.add(" ".join(w.title() for w in base).strip())
    return {x for x in al if x and len(x) >= 2}

# Must-have alias force list (helps with weird spellings)
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

# ---------------------- Catalog / Lookup ----------------------
LOOKUP = {}          # team_key -> {team, capacity, fill}
DISPLAY_NAME = {}    # team_key -> display string
TEAM_ALIASES = {}    # alias string -> team_key
TEAM_NAMES_ALL = []  # for UI datalist

def _build_from_catalog_json(path: str) -> bool:
    """Load a frozen catalog (recommended)."""
    if not os.path.exists(path):
        return False
    blob = json.load(open(path, "r"))
    teams = blob.get("teams", {})
    alias_to_key = blob.get("alias_to_key", {})
    public_names = blob.get("public_names", [])

    # Fill globals
    LOOKUP.clear()
    DISPLAY_NAME.clear()
    TEAM_ALIASES.clear()
    TEAM_NAMES_ALL.clear()

    for key, entry in teams.items():
        disp = entry.get("display") or key.title()
        DISPLAY_NAME[key] = disp
        LOOKUP[key] = {
            "team": disp,
            "capacity": entry.get("capacity"),
            "fill": entry.get("fill"),
        }
        TEAM_ALIASES[disp] = key
        for a in entry.get("aliases", []):
            TEAM_ALIASES.setdefault(a, key)

    # Include any provided alias map explicitly
    for a, k in alias_to_key.items():
        TEAM_ALIASES[a] = k

    # Make public list (fallback to all display names if not provided)
    if public_names:
        TEAM_NAMES_ALL.extend(public_names)
    else:
        TEAM_NAMES_ALL.extend(sorted(set(list(DISPLAY_NAME.values()) + list(TEAM_ALIASES.keys()))))

    # Force-add must-have aliases (mapped to best key by token overlap)
    def best_key_for(name: str):
        qtok = set(_tokens(name))
        best, score = None, -1
        for k, disp in DISPLAY_NAME.items():
            s = len(qtok & set(_tokens(disp)))
            if s > score:
                best, score = k, s
        return best

    for short, syns in MUST_HAVE_ALIASES.items():
        k = best_key_for(short)
        if k:
            TEAM_ALIASES[short] = k
            for s in syns:
                TEAM_ALIASES.setdefault(s, k)

    # Ensure dedup public names
    TEAM_NAMES_ALL[:] = sorted(set(TEAM_NAMES_ALL + list(TEAM_ALIASES.keys()) + list(DISPLAY_NAME.values())))
    return True

def _build_from_csv(path: str) -> bool:
    """Fallback: derive catalog from CSV (MAX capacity, MEDIAN fill) and add aliases."""
    if not os.path.exists(path):
        return False

    df = pd.read_csv(path)

    # Identify team-name columns
    name_like = [c for c in df.columns if any(k in c.lower() for k in ["team", "school", "program", "home team", "away team"])]
    if "Team" not in name_like and "Team" in df.columns:
        name_like.append("Team")

    # Choose canonical display by frequency
    display_counts = {}
    for col in name_like:
        if col not in df.columns:
            continue
        for val in df[col].dropna().astype(str):
            key = _norm_team(val)
            if not key:
                continue
            display_counts.setdefault(key, {})
            display_counts[key][val] = display_counts[key].get(val, 0) + 1

    DISPLAY_NAME.clear()
    for key, variants in display_counts.items():
        disp = sorted(variants.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[0][0]
        DISPLAY_NAME[key] = disp

    # Candidate capacity / fill
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

    # unify and aggregate
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

    agg = df.groupby("_team_key").agg(
        cap_max  = ("_cap_candidate", "max"),
        fill_med = ("_fill_candidate", "median")
    ).reset_index(drop=False)

    LOOKUP.clear()
    for _, r in agg.iterrows():
        key = r["_team_key"]
        if not key:
            continue
        disp = DISPLAY_NAME.get(key) or df.loc[df["_team_key"] == key, "_team_display_raw"].dropna().astype(str).iloc[0]
        cap = r["cap_max"]; fill = r["fill_med"]
        LOOKUP[key] = {
            "team": disp,
            "capacity": int(round(float(cap))) if pd.notna(cap) else None,
            "fill": float(fill) if pd.notna(fill) else None
        }

    # aliases
    TEAM_ALIASES.clear()
    for key, disp in DISPLAY_NAME.items():
        TEAM_ALIASES[disp] = key
        for a in _alias_variants(disp):
            TEAM_ALIASES.setdefault(a, key)

    # must-have aliases
    def best_key_for(name: str):
        qtok = set(_tokens(name))
        best, score = None, -1
        for k, disp in DISPLAY_NAME.items():
            s = len(qtok & set(_tokens(disp)))
            if s > score:
                best, score = k, s
        return best

    for short, syns in MUST_HAVE_ALIASES.items():
        k = best_key_for(short)
        if k:
            TEAM_ALIASES[short] = k
            for s in syns:
                TEAM_ALIASES.setdefault(s, k)

    # public names
    TEAM_NAMES_ALL.clear()
    TEAM_NAMES_ALL.extend(sorted(set(list(DISPLAY_NAME.values()) + list(TEAM_ALIASES.keys()))))
    return True

# Build data (prefer JSON, fallback to CSV)
if not _build_from_catalog_json(CATALOG_PATH):
    _build_from_csv(MERGED_CSV)

# Optional manual overrides example:
# OVERRIDES = {"notre dame": {"capacity": 80795}}
# for k, v in OVERRIDES.items():
#     if k in LOOKUP:
#         LOOKUP[k].update(v)

# ---------------------- Helpers ----------------------
def _best_key_for(query: str):
    """Exact key -> alias hit -> normalized alias -> token-overlap fuzzy best."""
    q = (query or "").strip()
    if not q:
        return None
    k = _norm_team(q)

    # Direct normalized display hit
    if k in DISPLAY_NAME:
        return k

    # Exact alias (case-sensitive) hit
    if q in TEAM_ALIASES:
        return TEAM_ALIASES[q]

    # Normalized alias map (build once)
    norm_alias_map = getattr(_best_key_for, "_norm_alias_map", None)
    if norm_alias_map is None:
        norm_alias_map = {}
        for alias, tkey in TEAM_ALIASES.items():
            norm_alias_map.setdefault(_norm_team(alias), tkey)
        _best_key_for._norm_alias_map = norm_alias_map
    if k in norm_alias_map:
        return norm_alias_map[k]

    # Fuzzy token overlap against display names
    qtok = set(_tokens(q))
    best, best_score = None, 0
    for tkey, disp in DISPLAY_NAME.items():
        dtok = set(_tokens(disp))
        score = len(qtok & dtok)
        if score > best_score:
            best, best_score = tkey, score
    return best

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
    if tkey and tkey in DISPLAY_NAME:
        info = {"team": DISPLAY_NAME[tkey]}
        if tkey in LOOKUP:
            cap = LOOKUP[tkey].get("capacity")
            fill = LOOKUP[tkey].get("fill")
            if cap is not None:  info["capacity"] = int(cap)
            if fill is not None: info["fill"] = max(0.0, min(float(fill), 1.0))
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
