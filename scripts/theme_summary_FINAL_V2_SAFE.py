# theme_summary_FINAL_V2_SAFE.py
# Stage 3: HDBSCAN cluster -> theme summary
# Fixes:
# 1) Uses actual numeric HDBSCAN "cluster" column.
# 2) Uses top representative papers by cluster_probability.
# 3) Avoids response_schema/AFC structured-output path.
# 4) Robustly extracts JSON from plain model text.
# 5) Every API attempt runs in a child process with a hard timeout.
# 6) Resume-safe; saves after each successful cluster.

import os, json, time, random, re, base64, subprocess, sys
from pathlib import Path
import pandas as pd

INPUT_FILE = "Deblurring_Clustered_1947.xlsx"
INPUT_SHEET = "Clustered Papers"
OUTPUT_FILE = "Deblurring_Theme_Summary_FINAL.xlsx"

MODEL = "gemma-4-26b-a4b-it"
REPRESENTATIVE_PAPERS = 15

MAX_ATTEMPTS = 3
HARD_TIMEOUT_SECONDS = 180
BASE_DELAY = 5.0

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not loaded in this kernel.")

PROMPT_TEMPLATE = r'''
You are performing thematic synthesis for an AI-assisted literature mapping workflow.

THESIS:
"Motion-Aware, Task-Driven Deblurring for Robust Visual Perception in Robotic Systems"

The thesis studies:
1. Image-only/blind motion deblurring baselines.
2. Motion-aware deblurring using robot/platform motion information such as gyro, IMU, inertial measurements or kinematic priors.
3. Whether image-restoration improvement translates into downstream robotic perception improvement.
4. Downstream tasks may include object detection, tracking, localization, SLAM, recognition, segmentation, OCR/inspection, or related machine-perception tasks.
5. Evaluation may include restoration metrics, task metrics, and computational efficiency.

You are given representative papers from ONE HDBSCAN semantic cluster.
The papers were selected by highest cluster-membership probability.

CLUSTER ID: {cluster_id}
TOTAL PAPERS IN CLUSTER: {paper_count}

REPRESENTATIVE PAPERS:
{papers}

Return ONLY one valid JSON object with EXACTLY these keys:
{{
  "theme": "concise technically specific cluster theme",
  "subthemes": ["..."],
  "research_questions": ["..."],
  "research_gaps": ["..."],
  "future_opportunities": ["..."],
  "key_methods": ["..."],
  "datasets_or_benchmarks": ["..."],
  "downstream_tasks": ["..."],
  "thesis_relevance": "2-4 concise sentences"
}}

RULES:
- Base the answer ONLY on the supplied paper information.
- Do not invent datasets, methods, conclusions, or gaps.
- Distinguish actual deblurring/restoration from papers that merely mention blur.
- Distinguish IMU/gyro/inertial deblurring from event-camera methods and generic sensor fusion.
- If a named dataset/benchmark is not explicitly evident, use [].
- Output JSON only. No markdown fences. No commentary.
'''

def clean_text(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).split())

def representative_subset(cluster_df, n=15):
    cols, asc = [], []
    if "cluster_probability" in cluster_df.columns:
        cols.append("cluster_probability"); asc.append(False)
    if "relevance" in cluster_df.columns:
        cols.append("relevance"); asc.append(False)
    if cols:
        cluster_df = cluster_df.sort_values(cols, ascending=asc, na_position="last")
    return cluster_df.head(n).copy()

def build_cluster_text(cluster_df):
    blocks = []
    for _, row in cluster_df.iterrows():
        blocks.append(
            "Title: " + clean_text(row.get("Title", "")) + "\n"
            "Abstract: " + clean_text(row.get("Abstract", "")) + "\n"
            "Author Keywords: " + clean_text(row.get("Author Keywords", "")) + "\n"
            "Index Keywords: " + clean_text(row.get("Index Keywords", "")) + "\n"
            "Screening Decision: " + clean_text(row.get("decision", "")) + "\n"
            "Screening Relevance: " + clean_text(row.get("relevance", "")) + "\n"
            "Detected Methods: " + clean_text(row.get("methods", "")) + "\n"
            "Screening Rationale: " + clean_text(row.get("reason", "")) + "\n"
            "Cluster Probability: " + clean_text(row.get("cluster_probability", "")) + "\n---"
        )
    return "\n".join(blocks)

def extract_json_object(text):
    if text is None:
        raise ValueError("Empty model response")
    t = str(text).strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)

    start = t.find("{")
    if start < 0:
        raise ValueError("No JSON object start found")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(t)):
        ch = t[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i+1])

    raise ValueError("No complete JSON object found")

REQUIRED_KEYS = [
    "theme", "subthemes", "research_questions", "research_gaps",
    "future_opportunities", "key_methods", "datasets_or_benchmarks",
    "downstream_tasks", "thesis_relevance"
]

def validate_result(obj):
    if not isinstance(obj, dict):
        raise ValueError("Model JSON is not an object")
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"Missing JSON keys: {missing}")
    for k in [
        "subthemes", "research_questions", "research_gaps",
        "future_opportunities", "key_methods", "datasets_or_benchmarks",
        "downstream_tasks"
    ]:
        if not isinstance(obj[k], list):
            raise ValueError(f"{k} must be a list")
    for k in ['theme', 'thesis_relevance']:
        if not isinstance(obj[k], str) or not obj[k].strip():
            raise ValueError(f'{k} must be a nonempty string')
    for k in REQUIRED_KEYS:
        if isinstance(obj[k], list) and any(not isinstance(item, str) for item in obj[k]):
            raise ValueError(f'{k} entries must be strings')
    return obj

CHILD_CODE = r'''
import os, sys, base64
from google import genai
payload_b64 = sys.stdin.read().strip()
prompt = base64.b64decode(payload_b64.encode("ascii")).decode("utf-8")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
resp = client.models.generate_content(
    model=os.environ["THEME_MODEL"],
    contents=prompt
)
sys.stdout.write(resp.text or "")
'''

def call_model_with_timeout(prompt):
    env = os.environ.copy()
    env["THEME_MODEL"] = MODEL
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    payload = base64.b64encode(prompt.encode("utf-8")).decode("ascii")

    cp = subprocess.run(
        [sys.executable, "-c", CHILD_CODE],
        input=payload,
        text=True,
        capture_output=True,
        timeout=HARD_TIMEOUT_SECONDS,
        env=env,
        encoding="utf-8",
        errors="replace"
    )

    if cp.returncode != 0:
        err = (cp.stderr or "").strip()
        raise RuntimeError(err[-2000:] if err else f"Child process exit code {cp.returncode}")

    return cp.stdout

def generate_summary(cluster_id, paper_count, paper_text):
    prompt = PROMPT_TEMPLATE.format(
        cluster_id=cluster_id,
        paper_count=paper_count,
        papers=paper_text
    )

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = call_model_with_timeout(prompt)
            return validate_result(extract_json_object(raw))

        except subprocess.TimeoutExpired:
            last_error = f"TIMEOUT after {HARD_TIMEOUT_SECONDS}s"
            print(f"   -> attempt {attempt}/{MAX_ATTEMPTS}: {last_error}")

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"   -> attempt {attempt}/{MAX_ATTEMPTS} failed: {last_error[:500]}")

        if attempt < MAX_ATTEMPTS:
            wait = BASE_DELAY * attempt + random.uniform(0, 1)
            print(f"      retrying in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(last_error or "Unknown theme-summary failure")

def as_excel_text(v):
    if isinstance(v, list):
        return " | ".join(str(x) for x in v)
    return v

def save_output(source_df, summaries):
    summary_df = pd.DataFrame(summaries)
    if len(summary_df):
        summary_df = summary_df.sort_values("cluster").reset_index(drop=True)

    rep_rows = []
    for row in summaries:
        c = int(row["cluster"])
        subset = source_df[source_df["cluster"] == c].copy()
        reps = representative_subset(subset, REPRESENTATIVE_PAPERS)
        for rank, (_, p) in enumerate(reps.iterrows(), start=1):
            rep_rows.append({
                "cluster": c,
                "rank_in_cluster": rank,
                "screen_id": p.get("screen_id", ""),
                "Title": p.get("Title", ""),
                "Year": p.get("Year", ""),
                "DOI": p.get("DOI", ""),
                "decision": p.get("decision", ""),
                "relevance": p.get("relevance", ""),
                "screening_theme": p.get("theme", ""),
                "cluster_probability": p.get("cluster_probability", ""),
                "Abstract": p.get("Abstract", ""),
                "Author Keywords": p.get("Author Keywords", ""),
                "methods": p.get("methods", ""),
                "reason": p.get("reason", ""),
            })

    cluster_qc = []
    cluster_ids = sorted(
        int(x) for x in pd.Series(source_df["cluster"]).dropna().unique()
        if int(x) >= 0
    )
    for c in cluster_ids:
        g = source_df[source_df["cluster"] == c]
        cluster_qc.append({
            "cluster": c,
            "paper_count": len(g),
            "KEEP": int((g["decision"] == "KEEP").sum()) if "decision" in g.columns else None,
            "MAYBE": int((g["decision"] == "MAYBE").sum()) if "decision" in g.columns else None,
            "mean_relevance": round(float(pd.to_numeric(g["relevance"], errors="coerce").mean()), 3)
                if "relevance" in g.columns else None,
            "mean_cluster_probability": round(float(pd.to_numeric(g["cluster_probability"], errors="coerce").mean()), 3)
                if "cluster_probability" in g.columns else None,
        })

    temp_output = Path(str(OUTPUT_FILE) + '.tmp.xlsx')
    with pd.ExcelWriter(temp_output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Theme Summary", index=False)
        pd.DataFrame(rep_rows).to_excel(writer, sheet_name="Representative Papers", index=False)
        pd.DataFrame(cluster_qc).to_excel(writer, sheet_name="Cluster QC", index=False)
    os.replace(temp_output, OUTPUT_FILE)

def main():
    print("=== STAGE 3 V2 SAFE: HDBSCAN CLUSTER -> THEME SUMMARY ===")
    print("Input:", INPUT_FILE)
    print("Model:", MODEL)
    print("Hard timeout:", HARD_TIMEOUT_SECONDS, "seconds per attempt")

    df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)

    required = ["cluster", "Title", "Abstract"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    from checkpoint_guard import bind_checkpoint
    bind_checkpoint('themes', INPUT_FILE, __file__, [OUTPUT_FILE])
    cluster_values = sorted(
        int(c)
        for c in pd.Series(df["cluster"]).dropna().unique()
        if int(c) >= 0
    )

    print("HDBSCAN clusters to summarize:", len(cluster_values))
    print("Cluster IDs:", cluster_values)

    summaries = []
    completed = set()

    if Path(OUTPUT_FILE).exists():
        try:
            existing = pd.read_excel(OUTPUT_FILE, sheet_name="Theme Summary")
            if "cluster" in existing.columns and len(existing):
                summaries = existing.to_dict("records")
                completed = set(int(x) for x in existing["cluster"].dropna().tolist())
                print(f"Resume mode: {len(completed)} clusters already summarized.")
            else:
                print("Existing output has no completed clusters; starting fresh.")
        except Exception as exc:
            raise RuntimeError('Existing theme checkpoint is unreadable; preserve it and use a fresh run directory') from exc
    else:
        print("Starting fresh.")

    for pos, cluster_id in enumerate(cluster_values, start=1):
        if cluster_id in completed:
            continue

        subset = df[df["cluster"] == cluster_id].copy()
        reps = representative_subset(subset, REPRESENTATIVE_PAPERS)

        print(
            f"\n[{pos}/{len(cluster_values)}] Cluster {cluster_id} | "
            f"papers={len(subset)} | representatives={len(reps)}"
        )

        result = generate_summary(
            cluster_id=cluster_id,
            paper_count=len(subset),
            paper_text=build_cluster_text(reps)
        )

        row = {
            "cluster": cluster_id,
            "paper_count": len(subset),
            "representative_paper_count": len(reps),
            "mean_relevance": round(float(pd.to_numeric(subset["relevance"], errors="coerce").mean()), 3)
                if "relevance" in subset.columns else None,
            "mean_cluster_probability": round(float(pd.to_numeric(subset["cluster_probability"], errors="coerce").mean()), 3)
                if "cluster_probability" in subset.columns else None,
            "KEEP_count": int((subset["decision"] == "KEEP").sum())
                if "decision" in subset.columns else None,
            "MAYBE_count": int((subset["decision"] == "MAYBE").sum())
                if "decision" in subset.columns else None,
            "theme": result["theme"],
            "subthemes": as_excel_text(result["subthemes"]),
            "research_questions": as_excel_text(result["research_questions"]),
            "research_gaps": as_excel_text(result["research_gaps"]),
            "future_opportunities": as_excel_text(result["future_opportunities"]),
            "key_methods": as_excel_text(result["key_methods"]),
            "datasets_or_benchmarks": as_excel_text(result["datasets_or_benchmarks"]),
            "downstream_tasks": as_excel_text(result["downstream_tasks"]),
            "thesis_relevance": result["thesis_relevance"],
            "representative_screen_ids": " | ".join(reps["screen_id"].astype(str).tolist())
                if "screen_id" in reps.columns else "",
            "representative_titles": " | ".join(reps["Title"].astype(str).tolist()),
            "status": "SUCCESS",
            "error": "",
        }

        summaries.append(row)
        completed.add(cluster_id)
        save_output(df, summaries)

        print("   -> SUCCESS:", result["theme"])
        time.sleep(1)

    save_output(df, summaries)
    print("\n=== THEME SUMMARY COMPLETE ===")
    print("Clusters summarized:", len(summaries))
    print("Output:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
