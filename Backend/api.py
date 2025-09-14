# api.py
from pathlib import Path
from typing import List, Optional, Dict, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from simple import run_suite_to_json, DB_NAME, parse_ratios_str

app = FastAPI(title="AQP Benchmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"ok": True, "db": DB_NAME}


@app.get("/init")
def init_api(
    dataset_filename: str = Query("yellow_tripdata_2015-01.csv"),
    table_name: str = Query("trips"),
    stratify_col: str = Query("RateCodeID"),
    hash_col: str = Query("payment_type"),
    numeric_col: str = Query("total_amount"),
    ratios: str = Query("0.01,0.02,0.05,0.1"),
    confidence: float = Query(0.95),
    verbose: bool = Query(False),
    group_by: Optional[str] = Query(None),
):
    """
    Run the pipeline with built-in CSV. Optionally include a single GROUP BY demo via ?group_by=RateCodeID
    """
    print(f"--- /init: dataset='{dataset_filename}', table='{table_name}' ---", flush=True)
    try:
        ratio_list = parse_ratios_str(ratios)
        payload = run_suite_to_json(
            dataset_filename=dataset_filename,
            table_name=table_name,
            stratify_col=stratify_col,
            hash_col=hash_col,
            numeric_col=numeric_col,
            ratios=ratio_list,
            confidence=confidence,
            verbose=verbose,
            group_by_col=group_by or None,
        )
        if not payload.get("ok"):
            raise HTTPException(status_code=400, detail=payload.get("error", "Unknown error"))
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ProcessRequest(BaseModel):
    dataset_id: Optional[str] = None
    sample_percentages: Optional[Union[List[int], str]] = None  # accepts [5,7,10] or "5,7,10"
    sum_column: Optional[str] = None
    avg_column: Optional[str] = None
    group_by_column: Optional[str] = None
    stratify_column: Optional[str] = "RateCodeID"


def _normalize_sample_percentages(raw: Optional[Union[List[int], str]]) -> List[int]:
    """Accept [5,7,10] or '5,7,10', return a cleaned list of ints."""
    if raw is None:
        return [5, 7, 10, 15, 20]
    if isinstance(raw, list):
        out: List[int] = []
        for v in raw:
            try:
                n = int(v)
                if n > 0:
                    out.append(n)
            except Exception:
                continue
        return out or [5, 7, 10, 15, 20]
    # string
    parts = [p.strip() for p in str(raw).split(",")]
    out = []
    for p in parts:
        if not p:
            continue
        try:
            n = int(p)
            if n > 0:
                out.append(n)
        except Exception:
            continue
    return out or [5, 7, 10, 15, 20]


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    save_path = UPLOAD_DIR / file.filename
    with save_path.open("wb") as f:
        f.write(await file.read())

    try:
        df_preview = pd.read_csv(save_path, nrows=200)
        cols = [{"name": str(c), "dtype": str(df_preview[c].dtype)} for c in df_preview.columns]
        df_rows = pd.read_csv(save_path, usecols=[df_preview.columns[0]])
        row_count = int(df_rows.shape[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    print(f"--- /upload_csv: saved to '{save_path}', rows≈{row_count} ---", flush=True)
    return {"dataset_id": str(save_path), "columns": cols, "row_count": row_count}


@app.post("/process")
def process(req: ProcessRequest):
    # 1) resolve dataset
    dataset_file: Optional[Path] = None
    if req.dataset_id:
        cand = Path(req.dataset_id)
        if cand.exists():
            dataset_file = cand
        else:
            alt = Path.cwd() / req.dataset_id
            if alt.exists():
                dataset_file = alt
    if dataset_file is None:
        fallback = Path("yellow_tripdata_2015-01.csv")
        if not fallback.exists():
            raise HTTPException(
                status_code=400,
                detail="No dataset_id provided (or not found) and default CSV not found: yellow_tripdata_2015-01.csv",
            )
        dataset_file = fallback

    # 2) sample percentages -> ratios
    samples = _normalize_sample_percentages(req.sample_percentages)
    ratios = [max(0.0, min(1.0, s / 100.0)) for s in samples]

    # 3) numeric column preference (for COUNT+AVG and SUM)
    numeric_col = req.avg_column or req.sum_column or "total_amount"

    # 4) run pipeline (optionally with a grouped example if group_by_column provided)
    print(
        f"--- /process: dataset='{dataset_file}', numeric_col='{numeric_col}', "
        f"samples={samples}, stratify='{req.stratify_column}', group_by='{req.group_by_column}' ---",
        flush=True,
    )
    payload = run_suite_to_json(
    dataset_filename=str(dataset_file),
    table_name="trips",
    stratify_col=req.stratify_column or "RateCodeID",
    hash_col="",  # <- don't force 'payment_type' for arbitrary uploads
    numeric_col=numeric_col,
    ratios=ratios,
    confidence=0.95,
    verbose=True,
    group_by_col=req.group_by_column or None,
)

    if not payload.get("ok"):
        raise HTTPException(status_code=400, detail=payload.get("error", "Unknown error"))

    # 5) reshape to {values, times}
    results = payload.get("results", [])
    values: Dict[str, Dict[str, List[Optional[float]]]] = {}
    times: Dict[str, Dict[str, List[int]]] = {}

    def ensure(metric: str):
        if metric not in values:
            values[metric] = {"Exact": [], "Simple": []}
        if metric not in times:
            times[metric] = {"Exact": [], "Simple": []}

    for block in results:
        qmap = {q["name"]: q for q in block.get("queries", [])}

        # Query 1: COUNT + AVG
        q1 = qmap.get("COUNT + AVG")
        if q1:
            exact_rows = q1.get("exact", {}).get("rows", []) or [{}]
            exact_time = q1.get("exact", {}).get("time_sec", 0.0)
            approx_rows = q1.get("approx", {}).get("rows", []) or [{}]
            approx_time = q1.get("approx", {}).get("time_sec", 0.0)

            ensure("total_trips")
            values["total_trips"]["Exact"].append(exact_rows[0].get("total_trips"))
            values["total_trips"]["Simple"].append(
                approx_rows[0].get("total_trips_estimate") if isinstance(approx_rows[0], dict) else None
            )
            times["total_trips"]["Exact"].append(int(exact_time * 1000))
            times["total_trips"]["Simple"].append(int(approx_time * 1000))

            ensure("avg_revenue")
            values["avg_revenue"]["Exact"].append(exact_rows[0].get("avg_revenue"))
            approx_avg = None
            if isinstance(approx_rows[0], dict):
                approx_avg = approx_rows[0].get("avg_revenue_estimate")
                if approx_avg is None:
                    s = approx_rows[0].get("__v_sum_avg_revenue", 0)
                    c = approx_rows[0].get("__v_count_avg_revenue", 0) or 0
                    approx_avg = (s / c) if c else None
            values["avg_revenue"]["Simple"].append(approx_avg)
            times["avg_revenue"]["Exact"].append(int(exact_time * 1000))
            times["avg_revenue"]["Simple"].append(int(approx_time * 1000))

        # Query 2: SUM
        q2 = qmap.get("SUM")
        if q2:
            exact_rows = q2.get("exact", {}).get("rows", []) or [{}]
            exact_time = q2.get("exact", {}).get("time_sec", 0.0)
            approx_rows = q2.get("approx", {}).get("rows", []) or [{}]
            approx_time = q2.get("approx", {}).get("time_sec", 0.0)

            ensure("total_revenue")
            values["total_revenue"]["Exact"].append(exact_rows[0].get("total_revenue"))
            values["total_revenue"]["Simple"].append(
                approx_rows[0].get("total_revenue_estimate") if isinstance(approx_rows[0], dict) else None
            )
            times["total_revenue"]["Exact"].append(int(exact_time * 1000))
            times["total_revenue"]["Simple"].append(int(approx_time * 1000))

    resp: Dict[str, Any] = {
        "values": values,
        "times": times,
        "meta": {"sample_percentages": samples},
    }

    # 6) include grouped example if present (frontend will render a small table)
    if payload.get("grouped_example"):
        resp["grouped_example"] = payload["grouped_example"]

    print(f"--- /process: responding with metrics={list(values.keys())}, grouped={bool(payload.get('grouped_example'))} ---", flush=True)
    return resp
