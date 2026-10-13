import glob, os, pandas as pd, numpy as np, h3
from sklearn.neighbors import BallTree

FOLDER = "PAN_INDIA_res9_HEX_LEVEL_AAC"
STORES = "stores.csv"          # columns: lat,lng  (skip coverage if file absent)
LEVEL  = 7                     # 7 = ~400k hexes (Kepler-friendly), 8 = ~1.5M

# 1. merge 35 CSVs, dedupe
files = [f for f in glob.glob(f"{FOLDER}/**/*.csv", recursive=True) if "__MACOSX" not in f]
df = pd.concat([pd.read_csv(f, dtype={"hex9": str}) for f in files], ignore_index=True)
df = df.drop_duplicates("hex9")
print(len(files), "files,", len(df), "hexes")          # expect 35

# 2. roll up to LEVEL (keeps Kepler fast)
df[f"hex{LEVEL}"] = df.hex9.map(lambda h: h3.cell_to_parent(h, LEVEL))
g = (df.sort_values("hex_aac", ascending=False)
       .groupby(f"hex{LEVEL}")
       .agg(hex_aac=("hex_aac", "sum"),
            consolidated_city=("consolidated_city", "first"),
            state=("state", "first"))
       .reset_index())

# 3. centroids (for heatmap layer + distance)
ll = g[f"hex{LEVEL}"].map(h3.cell_to_latlng)
g["lat"] = ll.str[0]; g["lng"] = ll.str[1]

# 4. store coverage
if os.path.exists(STORES):
    s = pd.read_csv(STORES).dropna(subset=["lat", "lng"])
    tree = BallTree(np.radians(s[["lat", "lng"]].values), metric="haversine")
    g["dist_km"] = tree.query(np.radians(g[["lat", "lng"]].values), k=1)[0][:, 0] * 6371
    for r in (3, 5):
        g[f"in_{r}km"] = g.dist_km <= r
        print(f"{r}km coverage: {g.loc[g[f'in_{r}km'], 'hex_aac'].sum() / g.hex_aac.sum():.1%}")
    (g.groupby(["state", "consolidated_city"])
       .apply(lambda x: pd.Series({
           "aac": x.hex_aac.sum(),
           "cov_3km_%": x.loc[x.in_3km, "hex_aac"].sum() / x.hex_aac.sum() * 100,
           "cov_5km_%": x.loc[x.in_5km, "hex_aac"].sum() / x.hex_aac.sum() * 100}))
       .sort_values("aac", ascending=False).to_csv("city_coverage_summary.csv"))

g.to_csv("kepler_hex.csv", index=False)
