"""
dashboard_routes.py — Dashboard endpoints for the FastAPI backend.

Ports the KPI row + Predictive Maintenance tabs from dashboard.py.
Mounted onto the main FastAPI app in main.py.

Reads the SAME artifact files dashboard.py reads (models/, outputs/plots/,
outputs/dashboard_data/) — nothing is recomputed here, just re-served as
JSON (and static images) for React instead of rendered with Streamlit.
"""
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

import config

router = APIRouter(prefix="/api/dashboard")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
PLOTS_DIR = Path(os.environ.get("PLOTS_DIR", os.path.join("outputs", "plots")))
DASHBOARD_DATA_DIR = Path(os.environ.get("DASHBOARD_DATA_DIR", os.path.join("outputs", "dashboard_data")))

# Point these at your real raw CSVs (same two files dashboard.py uses).
HIST_CSV = config.DATA_RAW_DIR / "Concated_Final_Cleaned_Dataset.csv"
REC_CSV = config.DATA_RAW_DIR / "Cleaned_Output_FAA-SDR-2026.csv"

_hist_cache = None
_rec_cache = None


def _load_hist():
    global _hist_cache
    if _hist_cache is None:
        df = pd.read_csv(HIST_CSV, index_col=0, low_memory=False)
        df["DifficultyDate"] = pd.to_datetime(df["DifficultyDate"], errors="coerce")
        df["SubmissionDate"] = pd.to_datetime(df["SubmissionDate"], errors="coerce")
        df["Year"] = df["DifficultyDate"].dt.year
        _hist_cache = df
    return _hist_cache


def _load_rec():
    global _rec_cache
    if _rec_cache is None:
        df = pd.read_csv(REC_CSV, index_col=0, low_memory=False)
        df["DifficultyDate"] = pd.to_datetime(df["DifficultyDate"], errors="coerce")
        df["SubmissionDate"] = pd.to_datetime(df["SubmissionDate"], errors="coerce")
        df["Year"] = df["DifficultyDate"].dt.year
        _rec_cache = df
    return _rec_cache


def _compute_kpis(df: pd.DataFrame, label: str) -> dict:
    total = len(df)
    makes = int(df["AircraftMake"].nunique()) if "AircraftMake" in df.columns else None
    parts = int(df["PartName"].nunique()) if "PartName" in df.columns else None
    regions = int(df["ReceivingRegionCode"].nunique()) if "ReceivingRegionCode" in df.columns else None
    years = int(df["Year"].nunique()) if "Year" in df.columns else None

    avg_days = None
    if "DifficultyDate" in df.columns and "SubmissionDate" in df.columns:
        sub = pd.to_datetime(df["SubmissionDate"], errors="coerce", utc=True)
        diff = pd.to_datetime(df["DifficultyDate"], errors="coerce", utc=True)
        days = (sub - diff).dt.days.dropna()
        avg_days = round(float(days.mean()), 1) if len(days) else None

    return {
        "total_sdrs": total,
        "days_to_submit": avg_days,
        "aircraft_makes": makes,
        "unique_parts": parts,
        "faa_regions": regions,
        "years_covered": years,
        "label": label,
    }


@router.get("/kpis")
def get_kpis(dataset: str = "historical"):
    df = _load_hist() if dataset == "historical" else _load_rec()
    label = "2021-2025" if dataset == "historical" else "2026 YTD"
    return _compute_kpis(df, label)


@router.get("/training-results")
def get_training_results():
    """Ports render_training_results_tab() — reads 01_train_model.py's
    saved artifacts. Read-only, nothing is retrained here."""
    meta_path = MODEL_DIR / "feature_meta.json"
    metrics_path = MODEL_DIR / "model_comparison_metrics.csv"

    if not (meta_path.exists() and metrics_path.exists()):
        raise HTTPException(
            status_code=404,
            detail=f"No training artifacts found in {MODEL_DIR}/. Run 01_train_model.py first.",
        )

    with open(meta_path) as f:
        meta = json.load(f)
    metrics_df = pd.read_csv(metrics_path)

    best_model = meta["best_model"]
    best_row = metrics_df.loc[metrics_df["model"] == best_model].iloc[0]

    metric_cards = {
        "accuracy": float(best_row["cv_accuracy_mean"]),
        "precision": float(best_row["cv_precision_mean"]),
        "recall": float(best_row["cv_recall_mean"]),
        "f1_score": float(best_row["cv_f1_mean"]),
        "roc_auc": float(best_row["cv_roc_auc_mean"]),
    }

    # Which of the 9 standard plots + per-model feature-importance plots
    # actually exist, so the frontend only tries to load real images.
    plot_files = [
        "01_severity_label_distribution.png",
        "02_confusion_matrices.png",
        "03_roc_curves.png",
        "04_precision_recall_curves.png",
        "05_model_comparison_f1.png",
        "07_cv_f1_boxplot.png",
        "08_cv_multimetric_comparison.png",
        "09_prediction_distribution.png",
    ]
    available_plots = [p for p in plot_files if (PLOTS_DIR / p).exists()]
    feature_importance_plots = sorted(
        p.name for p in PLOTS_DIR.glob("06_feature_importance_*.png")
    )

    return {
        "best_model": best_model,
        "trained_rows": meta["trained_rows_final_fit"],
        "trained_date_range": meta["trained_date_range"],
        "metric_cards": metric_cards,
        "all_models_comparison": metrics_df.to_dict(orient="records"),
        "available_plots": available_plots,
        "feature_importance_plots": feature_importance_plots,
        "label_rule": meta.get("label_rule"),
    }


@router.get("/prediction-results")
def get_prediction_results():
    """Ports render_prediction_results_tab() — reads 02_predict_2026.py's
    saved artifacts. Read-only, nothing is re-scored here."""
    kpi_path = DASHBOARD_DATA_DIR / "summary_kpis.json"
    if not kpi_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No prediction artifacts found in {DASHBOARD_DATA_DIR}/. Run 02_predict_2026.py first.",
        )

    with open(kpi_path) as f:
        kpis = json.load(f)

    def _read(name):
        p = DASHBOARD_DATA_DIR / name
        return pd.read_csv(p).to_dict(orient="records") if p.exists() else []

    plot_files = [
        "01_2026_monthly_trend.png",
        "02_2026_top_parts_critical.png",
        "03_2026_top_models_critical.png",
        "04_2026_prediction_distribution.png",
        "05_2026_probability_histogram.png",
    ]
    available_plots = [p for p in plot_files if (PLOTS_DIR / p).exists()]

    return {
        "kpis": kpis,
        "available_plots": available_plots,
        "monthly": _read("monthly_trend.csv"),
        "by_part": _read("by_part_name.csv")[:15],
        "by_model": _read("by_aircraft_model.csv")[:15],
        "by_region": _read("by_region.csv"),
        "top_examples": _read("top_critical_examples.csv"),
    }
