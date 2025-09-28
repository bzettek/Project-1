# export_teams_catalog.py
import pandas as pd, json, re

CSV = "merged_attendance_dataset.csv"
OUT = "teams_catalog.json"

STOP = {"university","of","the","and","at"}
KEEP2 = {"state","tech","a&m","aandm","southern","western","eastern","northern","central","poly","city","international"}

def norm(s:str)->str:
    s=(s or "").strip().lower()
    s=s.replace("&","and")
    s=re.sub(r"\bst\.\b","state",s)
    s=re.sub(r"[^a-z0-9\s-]"," ",s)
    s=re.sub(r"\s+"," ",s)
    return s.strip()

def toks(s): return [t for t in norm(s).split() if t and t not in STOP]

def alias_variants(display:str):
    al=set(); disp=display.strip()
    if not disp: return al
    T=toks(disp)
    if not T: return al
    al.add(T[0].title())
    if len(T)>=2 and T[1] in KEEP2: al.add((T[0]+" "+T[1]).title())
    m=re.search(r"university of ([a-z\s]+)$", norm(disp))
    if m:
        x=m.group(1).strip().title()
        if x: al.add(x)
    al.add(re.sub(r"\s*\([^)]*\)\s*$","",disp).strip())
    if len(T)>=2 and disp.split()[-1].endswith("s"):
        base=[T[0]]
        if len(T)>=2 and T[1] in KEEP2: base.append(T[1])
        al.add(" ".join(w.title() for w in base).strip())
    return {a for a in al if a and len(a)>=2}

# Force short aliases for common teams
MUST={
 "Ohio State":["Ohio St.","The Ohio State University","Ohio State University"],
 "Oregon":["Oregon Ducks","University of Oregon"],
 "Michigan":["Michigan Wolverines","University of Michigan"],
 "Texas A&M":["Texas A and M","Texas A&M University"],
 "Ole Miss":["Mississippi","University of Mississippi"],
 "USC":["Southern California","University of Southern California"],
 "UCLA":["University of California Los Angeles","California-Los Angeles"],
 "Pitt":["Pittsburgh","University of Pittsburgh"],
 "NC State":["North Carolina State"],
 "UTSA":["Texas San Antonio","Texas at San Antonio"],
 "BYU":["Brigham Young","Brigham Young University"],
 "SMU":["Southern Methodist","Southern Methodist University"],
 "Arizona State":["Arizona St."],
 "Colorado State":["Colorado St."],
 "Fresno State":["Fresno St."],
 "Kansas State":["Kansas St."],
 "Oklahoma State":["Oklahoma St."],
 "San Diego State":["San Diego St."],
 "San Jose State":["San Jose St."],
 "Boise State":["Boise St."],
 "Washington State":["Washington St."],
 "Utah State":["Utah St."],
 "Ball State":["Ball St."],
 "Bowling Green":["Bowling Green State"],
 "Miami":["Miami (FL)","Miami (Florida)"],
 "Miami (OH)":["Miami (Ohio)"],
 "Virginia Tech":["VT"],
}

df=pd.read_csv(CSV)

# find team-like columns
name_like=[c for c in df.columns if any(k in c.lower() for k in ["team","school","program","home team","away team"])]
if "Team" not in name_like and "Team" in df.columns: name_like.append("Team")

def pick_team(row):
    for col in (["Team"]+[c for c in name_like if c!="Team"]):
        if col in row and pd.notna(row[col]):
            s=str(row[col]).strip()
            if s: return s
    return ""

# canonical display by most frequent variant
display_by_key={}
counts={}
for _,r in df.iterrows():
    v=pick_team(r)
    if not v: continue
    k=norm(v)
    counts.setdefault(k,{})
    counts[k][v]=counts[k].get(v,0)+1
for k, m in counts.items():
    disp=sorted(m.items(), key=lambda kv:(kv[1], len(kv[0])), reverse=True)[0][0]
    display_by_key[k]=disp

# capacity & fill candidates per row (very permissive)
cap_col="Stadium Capacity" if "Stadium Capacity" in df.columns else None
fill_col="Fill Rate" if "Fill Rate" in df.columns else None

def row_capacity(r):
    if cap_col and pd.notna(r.get(cap_col)): return float(r[cap_col])
    if "AutoCapacity" in r and pd.notna(r["AutoCapacity"]): return float(r["AutoCapacity"])
    if all(c in r for c in ["2022AvgAttendance","2022CapacityPct"]) and pd.notna(r["2022AvgAttendance"]) and pd.notna(r["2022CapacityPct"]):
        try:
            pct=float(str(r["2022CapacityPct"]).rstrip("%"))/100.0
            if pct>0: return float(r["2022AvgAttendance"])/pct
        except Exception: pass
    return None

def row_fill(r):
    v=None
    if fill_col and pd.notna(r.get(fill_col)): v=float(r[fill_col])
    elif "AutoFillRate" in r and pd.notna(r["AutoFillRate"]): v=float(r["AutoFillRate"])
    elif "FiveYearCapacityPct" in r and pd.notna(r["FiveYearCapacityPct"]):
        try: v=float(str(r["FiveYearCapacityPct"]).rstrip("%"))/100.0
        except Exception: v=None
    if v is not None and v>1.5: v=v/100.0
    if v is not None: v=max(0.0, min(v, 1.0))
    return v

df["_team_raw"]=df.apply(pick_team, axis=1)
df["_key"]=df["_team_raw"].map(norm)
df["_cap"]=df.apply(row_capacity, axis=1)
df["_fill"]=df.apply(row_fill, axis=1)

agg=df.groupby("_key").agg(cap_max=("_cap","max"), fill_med=("_fill","median")).reset_index()

# Build catalog
catalog={}
for _,r in agg.iterrows():
    k=r["_key"]; 
    if not k: continue
    disp=display_by_key.get(k) or k.title()
    cap=int(round(float(r["cap_max"]))) if pd.notna(r["cap_max"]) else None
    fill=float(r["fill_med"]) if pd.notna(r["fill_med"]) else None
    catalog[k]={"display":disp,"capacity":cap,"fill":fill,"aliases":sorted(alias_variants(disp))}

# add forced aliases
def best_key(name:str):
    qt=set(toks(name)); best=None; score=-1
    for key,entry in catalog.items():
        st=set(toks(entry["display"]))
        s=len(qt & st)
        if s>score: best, score = key, s
    return best

alias_to_key={}
for k,e in catalog.items():
    alias_to_key[e["display"]]=k
    for a in e["aliases"]: alias_to_key.setdefault(a,k)

for short, syns in MUST.items():
    k=best_key(short)
    if not k: continue
    catalog[k]["aliases"]=sorted(set(catalog[k]["aliases"]) | {short, *syns})
    alias_to_key[short]=k
    for s in syns: alias_to_key[s]=k

public_names=sorted(set([e["display"] for e in catalog.values()] + list(alias_to_key.keys())))

with open(OUT,"w") as f:
    json.dump({"teams":catalog,"alias_to_key":alias_to_key,"public_names":public_names}, f, indent=2)

print(f"Wrote {OUT} with {len(catalog)} teams and {len(public_names)} public names.")