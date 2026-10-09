# gemini_embedding_cluster_1947.py
# Robust literature embedding + UMAP + HDBSCAN pipeline for Guruvignesh's deblurring thesis corpus.

import os, time, json, math, traceback
from pathlib import Path

import numpy as np
import pandas as pd

from google import genai
from google.genai import types

INPUT_XLSX = "Deblurring_Clustering_Input_1947.xlsx"
INPUT_SHEET = "Clustering Input"

MODEL = "gemini-embedding-2"
EMBED_DIM = 768

EMBED_NPY = "GeminiEmbedding2_1947_768d.npy"
STATUS_JSON = "GeminiEmbedding2_1947_status.json"
FAILURES_CSV = "GeminiEmbedding2_1947_failures.csv"

OUTPUT_XLSX = "Deblurring_Clustered_1947.xlsx"

SECONDS_BETWEEN_REQUESTS = 0.8
MAX_RETRIES = 5
CHECKPOINT_EVERY = 20

UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS_CLUSTER = 10
UMAP_MIN_DIST = 0.05
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 5

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not loaded in this kernel.")

try:
    import umap
except Exception as e:
    raise RuntimeError("Missing package 'umap-learn'. Install once with: pip install umap-learn") from e

try:
    import hdbscan
except Exception as e:
    raise RuntimeError("Missing package 'hdbscan'. Install once with: pip install hdbscan") from e

print("=== GEMINI EMBEDDING 2 + UMAP + HDBSCAN ===")
print("Input:", INPUT_XLSX)
print("Model:", MODEL, "| embedding dim:", EMBED_DIM)

df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET)

required = ["screen_id", "Title", "embedding_text", "theme", "decision", "relevance"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df['screen_id'].isna().any() or df['screen_id'].duplicated().any():
    raise ValueError('screen_id must be unique and nonempty')
if df['embedding_text'].isna().any() or df['embedding_text'].astype(str).str.strip().eq('').any():
    raise ValueError('embedding_text must be nonempty for every record')
from checkpoint_guard import bind_checkpoint
bind_checkpoint('embedding', INPUT_XLSX, __file__, [EMBED_NPY, STATUS_JSON, OUTPUT_XLSX])
n = len(df)
print("Papers:", n)

# ---------- Resume-safe embedding store ----------
if Path(EMBED_NPY).exists():
    embeddings = np.load(EMBED_NPY)
    if embeddings.shape != (n, EMBED_DIM):
        raise ValueError(
            f"Existing {EMBED_NPY} has shape {embeddings.shape}, expected {(n, EMBED_DIM)}"
        )
    print("Loaded existing embedding checkpoint.")
else:
    embeddings = np.full((n, EMBED_DIM), np.nan, dtype=np.float32)
    np.save(EMBED_NPY, embeddings)
    print("Created new embedding checkpoint.")

if Path(STATUS_JSON).exists():
    status = json.loads(Path(STATUS_JSON).read_text(encoding="utf-8"))
else:
    status = {}

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def save_checkpoint():
    np.save(EMBED_NPY, embeddings)
    Path(STATUS_JSON).write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def embed_one(text):
    result = client.models.embed_content(
        model=MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBED_DIM
        ),
    )
    vec = result.embeddings[0].values
    if len(vec) != EMBED_DIM:
        raise ValueError(f"Unexpected embedding length: {len(vec)}")
    vec = np.asarray(vec, dtype=np.float32)
    if not np.isfinite(vec).all() or np.linalg.norm(vec) == 0:
        raise ValueError('Embedding must be finite and nonzero')
    return vec

done_at_start = int(np.sum(~np.isnan(embeddings).any(axis=1)))
print(f"Already embedded: {done_at_start}/{n}")

failures = []

for i, row in df.iterrows():
    if not np.isnan(embeddings[i]).any():
        continue

    sid = str(row["screen_id"])
    title = str(row["Title"])[:120]
    text = str(row["embedding_text"])

    print(f"[{i+1}/{n}] {sid}: {title}")

    success = False
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vec = embed_one(text)
            embeddings[i] = vec
            status[sid] = {"status": "SUCCESS", "attempts": attempt}
            success = True
            print("   -> SUCCESS")
            break

        except Exception as e:
            last_error = repr(e)
            wait = min(60, 2 ** attempt + 2)
            print(f"   -> attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}")
            if attempt < MAX_RETRIES:
                print(f"      waiting {wait}s before retry")
                time.sleep(wait)

    if not success:
        status[sid] = {"status": "ERROR", "error": last_error}
        failures.append({
            "screen_id": sid,
            "Title": row["Title"],
            "error": last_error,
        })
        print("   -> ERROR saved; will not silently substitute a vector.")

    if (i + 1) % CHECKPOINT_EVERY == 0 or not success:
        save_checkpoint()

    time.sleep(SECONDS_BETWEEN_REQUESTS)

save_checkpoint()

# Write failures file if any
if failures:
    pd.DataFrame(failures).to_csv(FAILURES_CSV, index=False)
    print(f"\nEmbedding pass finished with {len(failures)} failures.")
else:
    if Path(FAILURES_CSV).exists():
        Path(FAILURES_CSV).unlink()
    print("\nEmbedding pass finished with 0 failures.")

missing_mask = np.isnan(embeddings).any(axis=1)
missing_count = int(missing_mask.sum())

if missing_count:
    failed_ids = df.loc[missing_mask, "screen_id"].tolist()
    print("Unembedded papers:", missing_count)
    print("First failed IDs:", failed_ids[:20])
    raise RuntimeError(
        "Clustering stopped because some embeddings are missing. "
        "Rerun this same script later; it will resume only the missing rows."
    )

if not np.isfinite(embeddings).all() or (np.linalg.norm(embeddings, axis=1) == 0).any():
    raise ValueError('Invalid cached embedding; archive this run and start fresh')
print("All embeddings complete:", n)

# L2 normalization helps cosine-style semantic geometry.
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
emb_norm = embeddings / norms

# ---------- UMAP for clustering ----------
print("\nRunning UMAP for clustering...")
umap_cluster = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    n_components=UMAP_N_COMPONENTS_CLUSTER,
    min_dist=UMAP_MIN_DIST,
    metric="cosine",
    random_state=42,
)
z10 = umap_cluster.fit_transform(emb_norm)

# ---------- HDBSCAN ----------
print("Running HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
labels = clusterer.fit_predict(z10)
probs = clusterer.probabilities_

# ---------- 2D UMAP for visualization ----------
print("Running 2D UMAP for visualization...")
umap_2d = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    n_components=2,
    min_dist=0.10,
    metric="cosine",
    random_state=42,
)
z2 = umap_2d.fit_transform(emb_norm)

out = df.copy()
out["cluster"] = labels
out["cluster_probability"] = probs
out["umap_x"] = z2[:, 0]
out["umap_y"] = z2[:, 1]

cluster_sizes = out[out["cluster"] >= 0]["cluster"].value_counts().to_dict()
out["cluster_size"] = out["cluster"].map(cluster_sizes).fillna(0).astype(int)
out["is_noise"] = out["cluster"].eq(-1)

# Cluster summary
summary_rows = []
for c in sorted([x for x in out["cluster"].unique() if x >= 0]):
    g = out[out["cluster"] == c]
    themes = g["theme"].value_counts()
    decisions = g["decision"].value_counts()
    summary_rows.append({
        "cluster": int(c),
        "size": len(g),
        "KEEP": int(decisions.get("KEEP", 0)),
        "MAYBE": int(decisions.get("MAYBE", 0)),
        "dominant_screening_theme": themes.index[0] if len(themes) else "",
        "dominant_theme_count": int(themes.iloc[0]) if len(themes) else 0,
        "mean_relevance": round(float(g["relevance"].mean()), 3),
        "mean_cluster_probability": round(float(g["cluster_probability"].mean()), 3),
        "representative_titles": " | ".join(
            g.sort_values("cluster_probability", ascending=False)["Title"]
             .astype(str).head(5).tolist()
        ),
    })

summary = pd.DataFrame(summary_rows)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    out.to_excel(writer, sheet_name="Clustered Papers", index=False)
    summary.to_excel(writer, sheet_name="Cluster Summary", index=False)

n_clusters = len(summary)
n_noise = int((labels == -1).sum())

print("\n=== CLUSTERING COMPLETE ===")
print("Clusters:", n_clusters)
print("Noise/outliers:", n_noise, f"({n_noise/n:.1%})")
print("Output:", OUTPUT_XLSX)
print("Embeddings:", EMBED_NPY)
