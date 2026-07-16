"""
02_predict_2026.py
===================
Loads the production pipeline trained by 01_train_model.py and scores the
UNSEEN 2026 SDR data, then generates the row-level predictions + rollup
insight tables/visualizations that feed the teammate's dashboard.

Run 01_train_model.py first (it writes ./models/severity_pipeline.joblib +
./models/feature_meta.json). This script only reads those artifacts, it
never retrains anything.

Outputs (written to ./outputs/dashboard_data and ./outputs/plots):
    predictions_detail.csv        <- every 2026 row + predicted label/probability
    summary_kpis.json             <- headline numbers for a dashboard KPI strip
    monthly_trend.csv             <- critical count/rate by month
    by_aircraft_model.csv         <- critical count/rate by AircraftModel
    by_part_name.csv              <- critical count/rate by PartName
    by_region.csv                 <- critical count/rate by ReceivingRegionCode
    top_critical_examples.csv     <- highest-probability critical records
    outputs/plots/*.png           <- standalone visualizations (optional, for review)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sdr_common import to_dense, engineer_features  # noqa: F401 (to_dense needed for unpickling)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sns.set_style("whitegrid")

# =========================================
# 0. CONFIG / PATHS
# =========================================
MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "severity_pipeline.joblib")
META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")

# Edit this path (or set env var PREDICT_DATA_PATH) to point at your local file.
DATA_2026_PATH = os.environ.get(
    "PREDICT_DATA_PATH",
    r"C:\Users\Priyanka\Downloads\faa_sdr_genai_project\genai_project\data\raw\Cleaned_Output_FAA-SDR-2026.csv"
)

DASHBOARD_DIR = os.path.join("outputs", "dashboard_data")
PLOTS_DIR = os.path.join("outputs", "plots")
os.makedirs(DASHBOARD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

CRITICAL_PROB_THRESHOLD = 0.5  # decision threshold on predict_proba for the "Critical" label


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {path}")


# =========================================
# 1. LOAD TRAINED PIPELINE + METADATA
# =========================================
if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(
        f"'{PIPELINE_PATH}' not found. Run 01_train_model.py first to train and save the model."
    )

pipeline = joblib.load(PIPELINE_PATH)
with open(META_PATH) as f:
    meta = json.load(f)

cat_cols = meta["cat_cols"]
num_cols = meta["num_cols"]
text_col = meta["text_col"]
feature_cols = meta["feature_cols"]
print(f"Loaded pipeline (best model: {meta['best_model']}), "
      f"trained on {meta['trained_rows_final_fit']:,} historical rows "
      f"({meta['trained_date_range'][0]} to {meta['trained_date_range'][1]}).")

# =========================================
# 2. LOAD 2026 DATA
# =========================================
if not os.path.exists(DATA_2026_PATH):
    raise FileNotFoundError(
        f"2026 dataset not found at '{DATA_2026_PATH}'. Set the PREDICT_DATA_PATH "
        "env var or edit DATA_2026_PATH at the top of this script."
    )
df = pd.read_csv(DATA_2026_PATH)
print("Raw 2026 shape:", df.shape)

required_raw_cols = cat_cols + [text_col, "DifficultyDate"]
missing = [c for c in required_raw_cols if c not in df.columns]
if missing:
    raise ValueError(f"2026 dataset missing columns the model needs: {missing}")

# =========================================
# 3. FEATURE ENGINEERING (identical logic to training, imported shared code)
# =========================================
df = engineer_features(df, cat_cols, text_col, date_col="DifficultyDate")
X_2026 = df[feature_cols]

# =========================================
# 4. PREDICT
# =========================================
print("\nScoring 2026 records...")
proba = pipeline.predict_proba(X_2026)[:, 1]
pred = (proba >= CRITICAL_PROB_THRESHOLD).astype(int)

df["predicted_severity_prob"] = proba
df["predicted_severity_flag"] = pred
df["predicted_severity_label"] = np.where(pred == 1, "Critical", "Non-Critical")

n_critical = int(pred.sum())
n_total = len(df)
critical_rate = n_critical / n_total if n_total else 0.0
print(f"Predicted {n_critical:,} / {n_total:,} records as Critical ({critical_rate:.1%}).")

# =========================================
# 5. ROW-LEVEL EXPORT FOR THE DASHBOARD
# =========================================
export_cols = [
    "OperatorControlNumber", "DifficultyDate", "SubmissionDate", "OperatorDesignator",
    "ReceivingRegionCode", "ReceivingDistrictOffice", "SDRType", "JASCCode",
    "NatureOfConditionA", "StageOfOperationCode", "RegistryNNumber", "AircraftMake",
    "AircraftModel", "AircraftTotalTime", "AircraftTotalCycles", "PartName", "PartCondition",
    "Discrepancy", "predicted_severity_prob", "predicted_severity_flag", "predicted_severity_label",
]
export_cols = [c for c in export_cols if c in df.columns]
detail_path = os.path.join(DASHBOARD_DIR, "predictions_detail.csv")
df[export_cols].to_csv(detail_path, index=False)
print(f"\nSaved -> {detail_path}")

# =========================================
# 6. SUMMARY KPIs (for a dashboard header strip)
# =========================================
df["DifficultyDate"] = pd.to_datetime(df["DifficultyDate"], errors="coerce", utc=True)
kpis = {
    "total_records_2026": int(n_total),
    "predicted_critical_count": n_critical,
    "predicted_non_critical_count": int(n_total - n_critical),
    "predicted_critical_rate": round(critical_rate, 4),
    "avg_critical_probability": round(float(proba.mean()), 4),
    "date_range": [
        str(df["DifficultyDate"].min()), str(df["DifficultyDate"].max())
    ],
    "model_used": meta["best_model"],
    "model_trained_through": meta["trained_date_range"][1],
}
with open(os.path.join(DASHBOARD_DIR, "summary_kpis.json"), "w") as f:
    json.dump(kpis, f, indent=2)
print(f"Saved -> {os.path.join(DASHBOARD_DIR, 'summary_kpis.json')}")
print(json.dumps(kpis, indent=2))

# =========================================
# 7. MONTHLY TREND
# =========================================
df["year_month"] = df["DifficultyDate"].dt.to_period("M").astype(str)
monthly = df.groupby("year_month").agg(
    total_records=("predicted_severity_flag", "count"),
    critical_count=("predicted_severity_flag", "sum"),
).reset_index()
monthly["critical_rate"] = (monthly["critical_count"] / monthly["total_records"]).round(4)
monthly.to_csv(os.path.join(DASHBOARD_DIR, "monthly_trend.csv"), index=False)
print(f"Saved -> {os.path.join(DASHBOARD_DIR, 'monthly_trend.csv')}")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar(monthly["year_month"], monthly["total_records"], color="#B0C4DE", label="Total records")
ax1.bar(monthly["year_month"], monthly["critical_count"], color="#C44E52", label="Critical (predicted)")
ax1.set_ylabel("Record count")
ax1.set_xlabel("Month")
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
ax2 = ax1.twinx()
ax2.plot(monthly["year_month"], monthly["critical_rate"], color="black", marker="o", label="Critical rate")
ax2.set_ylabel("Critical rate")
ax2.set_ylim(0, 1)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.set_title("2026 Monthly Volume & Predicted Critical Rate")
save_fig(fig, "01_2026_monthly_trend.png")

# =========================================
# 8. BREAKDOWNS: AIRCRAFT MODEL / PART NAME / REGION
# =========================================
def rollup(df, group_col, top_n=20):
    g = df.groupby(group_col).agg(
        total_records=("predicted_severity_flag", "count"),
        critical_count=("predicted_severity_flag", "sum"),
        avg_critical_prob=("predicted_severity_prob", "mean"),
    ).reset_index()
    g["critical_rate"] = (g["critical_count"] / g["total_records"]).round(4)
    g["avg_critical_prob"] = g["avg_critical_prob"].round(4)
    return g.sort_values("critical_count", ascending=False).head(top_n)


by_model = rollup(df, "AircraftModel")
by_model.to_csv(os.path.join(DASHBOARD_DIR, "by_aircraft_model.csv"), index=False)
print(f"Saved -> {os.path.join(DASHBOARD_DIR, 'by_aircraft_model.csv')}")

by_part = rollup(df, "PartName")
by_part.to_csv(os.path.join(DASHBOARD_DIR, "by_part_name.csv"), index=False)
print(f"Saved -> {os.path.join(DASHBOARD_DIR, 'by_part_name.csv')}")

if "ReceivingRegionCode" in df.columns:
    by_region = rollup(df, "ReceivingRegionCode", top_n=50)
    by_region.to_csv(os.path.join(DASHBOARD_DIR, "by_region.csv"), index=False)
    print(f"Saved -> {os.path.join(DASHBOARD_DIR, 'by_region.csv')}")

fig, ax = plt.subplots(figsize=(8, 7))
top15_parts = by_part.head(15).sort_values("critical_count")
ax.barh(top15_parts["PartName"], top15_parts["critical_count"], color="#C44E52")
ax.set_xlabel("Predicted critical count")
ax.set_title("Top 15 Parts by Predicted Critical Volume (2026)")
save_fig(fig, "02_2026_top_parts_critical.png")

fig, ax = plt.subplots(figsize=(8, 7))
top15_models = by_model.head(15).sort_values("critical_count")
ax.barh(top15_models["AircraftModel"], top15_models["critical_count"], color="#4C72B0")
ax.set_xlabel("Predicted critical count")
ax.set_title("Top 15 Aircraft Models by Predicted Critical Volume (2026)")
save_fig(fig, "03_2026_top_models_critical.png")

# =========================================
# 9. TOP CRITICAL EXAMPLES (for a dashboard "watchlist" panel)
# =========================================
top_examples_cols = [c for c in [
    "OperatorControlNumber", "DifficultyDate", "AircraftMake", "AircraftModel",
    "PartName", "PartCondition", "predicted_severity_prob", "Discrepancy"
] if c in df.columns]
top_examples = df.sort_values("predicted_severity_prob", ascending=False)[top_examples_cols].head(100)
top_examples.to_csv(os.path.join(DASHBOARD_DIR, "top_critical_examples.csv"), index=False)
print(f"Saved -> {os.path.join(DASHBOARD_DIR, 'top_critical_examples.csv')}")

# =========================================
# 10. DISTRIBUTION / PROBABILITY VISUALS
# =========================================
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x="predicted_severity_label", data=df, ax=ax, palette=["#4C72B0", "#C44E52"])
ax.set_title("2026 Predicted Severity Distribution")
ax.set_xlabel("")
save_fig(fig, "04_2026_prediction_distribution.png")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(proba, bins=40, color="#4C72B0")
ax.axvline(CRITICAL_PROB_THRESHOLD, color="red", linestyle="--", label=f"Decision threshold ({CRITICAL_PROB_THRESHOLD})")
ax.set_xlabel("Predicted probability of Critical")
ax.set_ylabel("Record count")
ax.set_title("2026 Predicted Probability Distribution")
ax.legend()
save_fig(fig, "05_2026_probability_histogram.png")

print("\nDone. Dashboard-ready files are in ./outputs/dashboard_data, plots in ./outputs/plots.")
