# export_teams_catalog.py
import csv, json
from collections import OrderedDict

CSV = "update_tNames_cap_fill.csv"
OUT = "teams_catalog.json"


def _normalize(name: str) -> str:
    return (name or "").strip().lower()


def _parse_capacity(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _parse_fill(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        fill = float(text)
    except ValueError:
        return None
    if fill > 1.5:
        fill /= 100.0
    return max(0.0, min(fill, 1.0))


canonical = OrderedDict()
with open(CSV, newline="") as f:
    reader = csv.DictReader(f)
    required = {"Team", "Capacity", "Fill Rate"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        missing = required - set(reader.fieldnames or [])
        raise SystemExit(f"CSV missing columns: {', '.join(sorted(missing))}")

    for row in reader:
        team = (row.get("Team") or "").strip()
        if not team:
            continue
        cap = _parse_capacity(row.get("Capacity"))
        fill = _parse_fill(row.get("Fill Rate"))
        canonical[team] = {"capacity": cap, "fill": fill}

catalog = {}
alias_to_key = {}
public_names = []

for team, values in canonical.items():
    key = _normalize(team)
    if not key:
        continue
    catalog[key] = {
        "display": team,
        "capacity": values["capacity"],
        "fill": values["fill"],
        "aliases": []
    }
    alias_to_key[team] = key
    public_names.append(team)

public_names = sorted(set(public_names))

with open(OUT, "w") as f:
    json.dump({
        "teams": catalog,
        "alias_to_key": alias_to_key,
        "public_names": public_names
    }, f, indent=2)

print(f"Wrote {OUT} with {len(catalog)} teams.")
