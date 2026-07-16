"""
01_train_model.py
==================
Trains an SDR "severity" classifier on the HISTORICAL FAA dataset (2021-2025)
and persists the fitted production pipeline + metadata for later scoring of
the unseen 2026 data (see 02_predict_2026.py).

Three algorithms are trained and compared:
    1. Logistic Regression      (linear baseline)
    2. Random Forest            (bagged trees)
    3. XGBoost                  (boosted trees, recommended)
       -> falls back automatically to sklearn's HistGradientBoostingClassifier
          (fit on a stratified subsample, since it needs dense input) if
          xgboost isn't installed. For best results:  pip install xgboost

Validation:
    - Single 80/20 holdout split -> classification report + confusion matrix
      + ROC/PR curves for all 3 models
    - 5-fold Stratified CV, F1 only              (cross_val_score)
    - 5-fold Stratified CV, multi-metric         (cross_validate: accuracy,
      precision, recall, f1, roc_auc)

Outputs (written to ./models and ./outputs/plots):
    models/severity_pipeline.joblib      <- best model, refit on 100% of data
    models/feature_meta.json             <- feature config + label rules + metrics
    models/model_comparison_metrics.csv  <- holdout + CV metrics for all 3 models
    outputs/plots/*.png                  <- all visualizations
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no plt.show() needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sdr_common import to_dense, engineer_features
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score, roc_auc_score
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
sns.set_style("whitegrid")

# =========================================
# 0. CONFIG / PATHS
# =========================================
# Edit this path (or set env var TRAIN_DATA_PATH) to point at your local file.
DATA_PATH = os.environ.get(
    "TRAIN_DATA_PATH",
    r"C:\Users\Priyanka\Downloads\faa_sdr_genai_project\genai_project\data\raw\Concated_Final_Cleaned_Dataset.csv"
)

MODEL_DIR = "models"
PLOTS_DIR = os.path.join("outputs", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

REQUIRED_COLS = [
    "PartCondition", "Discrepancy", "DifficultyDate",
    "AircraftModel", "PartName", "NatureOfConditionA", "StageOfOperationCode"
]


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {path}")


# =========================================
# 1. LOAD HISTORICAL (2021-2025) DATA
# =========================================
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Historical dataset not found at '{path}'. Set the TRAIN_DATA_PATH "
            "env var or edit DATA_PATH at the top of this script."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


df = load_data(DATA_PATH)
print("Raw historical shape:", df.shape)

# NOTE: SubmissionDate has mixed timezone-aware/naive string formats which
# makes pandas coerce a large chunk of it to NaT if parsed naively.
# DifficultyDate is clean (0 NaT, evenly covers 2021-2025) so it is used as
# the canonical date field throughout both this script and the predictor.
df["_check_date"] = pd.to_datetime(df["DifficultyDate"], errors="coerce", utc=True)
n_future = (df["_check_date"].dt.year >= 2026).sum()
n_bad = df["_check_date"].isna().sum()
if n_bad > 0:
    print(f"WARNING: {n_bad} rows have an unparseable DifficultyDate and will get default time features.")
if n_future > 0:
    print(f"WARNING: {n_future} rows have DifficultyDate in 2026+. These should NOT be in the training set.")
df = df.drop(columns=["_check_date"])

# =========================================
# 2. SEVERITY LABEL
# =========================================
# CRITICAL   -> airworthiness-significant findings (severity_flag = 1)
# NON-EVENT  -> no actual discrepancy occurred, dropped from training entirely
# everything else -> non-critical (severity_flag = 0), with a light text
#                     fallback (rare) for PartCondition values not seen below
critical_conditions = {
    "CRACKED", "INOPERATIVE", "DAMAGED", "FAULTED", "LOW PRESSURE", "ODOR", "FAILED", "BROKEN",
    "LOOSE", "EXCESS PLAY", "MISSING", "LEAKING", "WORN", "DENTED", "OUT OF ADJUST", "CHAFED", "GOUGED",
    "FAULTY", "DELAMINATED", "DETACHED", "MALFUNCTIONED", "UNSECURE", "FALSE ACTIVATION", "DEFECTIVE",
    "PUNCTURED", "LACK OF LUBE", "PULLED", "CONTAMINATED", "ILLUMINATED",
    "DISCONNECTED", "OUT OF POSITION", "DISCHARGED",
    "CORRODED", "RUSTED",  # CORRODED alone ~22% of all rows; airworthiness-significant,
                            # previously fell through to the weak text fallback unlabeled.
}
non_event_conditions = {"NO DISCREPANCY", "NOT REPORTED", "NONE", "UNKNOWN", "NO TEST"}
non_critical_conditions = {
    'DIRTY', 'DEBONDED', 'WARNING MESSAGE', 'BINDING', 'FOD', 'BURNED OUT', 'WRONG PART', 'SMOKE', 'STUCK',
    'DETERIORATED', 'ABNORMAL', 'STICKING', 'DISLODGED', 'MISINSTALLED', 'BENT', 'SCRATCHED', 'INTERMITTENT',
    'UNWANTED DEPLOY', 'OUT OF LIMITS', 'FALSE INDICATION', 'VIBRATION', 'UNSERVICEABLE', 'CLOGGED', 'SHEARED',
    'SHORTED', 'OPEN', 'DEFORMED', 'NOT SEATED', 'MECHANIC ERROR', 'BURNED', 'TORN', 'OUT OF TOLERANCE', 'BLOWN',
    'ARCED', 'ELONGATED', 'FIRE', 'FUMES', 'JAMMED', 'TRIPPED', 'GROUND DAMAGE', 'STALLED', 'MIGRATED', 'OVERHEATED',
    'OVERSERVICED', 'WEAK', 'UNPACKED', 'OVERTEMP', 'PEELING', 'UNRAVELED', 'EXPIRED', 'FRAYED',
    'OUT OF ALIGNMENT', 'SEIZED', 'OBSTRUCTED', 'FAILED INSP', 'FOM', 'MISREPAIRED', 'OUT OF RIG', 'SEPARATED',
    'BIRD INGESTION', 'BULGED', 'DIM', 'DISPLACED', 'READS LOW', 'DEGRADED', 'DISENGAGED', 'LOCKED', 'MAKING METAL',
    'OUT OF BALANCE', 'POWER LOSS', 'ACTIVATED', 'FAULT MESSAGE', 'INFLT SEPARATION', 'INSTALLED', 'MELTED', 'MISWIRED',
    'NICKED', 'NOISY', 'PARTIAL DEPLOY', 'BRINELLED', 'COLLAPSED', 'DEMODIFICATION', 'EMPTY', 'FROZEN', 'LOW',
    'NO INDICATION', 'NOISE', 'NOT OPENED', 'SPARKS', 'STIFF', 'WARNING LIGHT', 'ACTIVATION ERROR', 'BURST', 'CUT',
    'EXPOSED', 'INCORRECT', 'INDICATION', 'INTERFERENCE', 'RUPTURED', 'SHUTDOWN', 'SPLIT', 'TEMP REPAIR', 'UNLOCKED',
    'ASYMETRIC', 'BUCKLED', 'CANNING', 'CRAZED', 'DEPLOYED', 'ERRATIC', 'FRACTURED', 'MISCONFIGURED',
    'MISMANUFACTURED', 'UNCONTROLLABLE', 'BACKED OUT', 'BLEW OUT', 'BLOCKED', 'DESTROYED', 'ERODED', 'FADED',
    'FLUCTUATES', 'MISALIGNED', 'OIL CONSUMPTION', 'OVERPRESSURED', 'PINCHED', 'UNLATCHED', 'AUTO SHUTDOWN',
    'BURRED', 'CHIPPED', 'CLOSED', 'CORRUPTED', 'DEPARTED', 'DEPLETED', 'DRAGGING', 'FLAMED OUT', 'FLAT', 'FOULED',
    'FULL', 'HAZE', 'INADEQUATE', 'KINKED', 'MISOVERHAULED', 'MISROUTED', 'MULTIPLE FAIL', 'NO DEPLOY',
    'NOT CLOSED', 'NOT ENTERED', 'OFF TRACK', 'OUT OF RANGE', 'OVERSPEED', 'PERFORATED', 'REVERSED', 'SLOW',
    'STICKS', 'UNSAFETIED', 'ACTIVE', 'BLISTERED', 'CAVITATES', 'CHATTERING', 'COKED', 'CREW ERROR', 'CRUSHED',
    'DEFLATED', 'DEFLECTION', 'EXPLODED', 'GALLED', 'GROOVED', 'HIGH PRESSURE', 'HUNG START', 'ICED', 'INACCURATE',
    'INACTIVE', 'LOST', 'LOW QUANTITY', 'LOW TENSION', 'MISLOCATED', 'NOT REQUIRED', 'OSCILLATES', 'OUT OF SEQUENCE',
    'OUT OF TRIM', 'OVERTORQUED', 'PILOT ERROR', 'PRESSURE LOSS', 'RESTRICTED', 'RUNAWAY', 'SCORED', 'SEVERED',
    'SHINGLED', 'SLIPPED', 'SOILED', 'SPINNING', 'STRIPPED', 'SURGES', 'THERMAL RUNAWAY', 'TIGHT', 'TWISTED',
    'UNBONDED', 'UNDERSERVICED', 'UNDERTORQUED', 'UNRELIABLE', 'UNRESPONSIVE', 'UNSTABLE', 'WRINKLED'
}
TEXT_FALLBACK_KEYWORDS = ["failure", "fire", "smoke", "leak", "crack", "engine issue"]


def assign_severity(cond, discrepancy_text):
    cond = str(cond).upper().strip()
    if cond in non_event_conditions:
        return None
    if cond in critical_conditions:
        return 1
    if cond in non_critical_conditions:
        return 0
    text = str(discrepancy_text).lower()
    return 1 if any(k in text for k in TEXT_FALLBACK_KEYWORDS) else 0


print("\nAssigning severity labels...")
df["severity_flag"] = df.apply(lambda r: assign_severity(r["PartCondition"], r["Discrepancy"]), axis=1)
n_dropped = df["severity_flag"].isna().sum()
df = df.dropna(subset=["severity_flag"]).copy()
df["severity_flag"] = df["severity_flag"].astype(int)
print(f"Dropped {n_dropped} non-event/unlabeled rows. Training shape: {df.shape}")
label_dist = df["severity_flag"].value_counts(normalize=True).sort_index()
print(label_dist)

fig, ax = plt.subplots(figsize=(5, 4))
counts = df["severity_flag"].value_counts().sort_index()
ax.bar(["Non-Critical (0)", "Critical (1)"], counts.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom")
ax.set_title("Severity Label Distribution (2021-2025)")
ax.set_ylabel("Record count")
save_fig(fig, "01_severity_label_distribution.png")

# =========================================
# 3. FEATURE ENGINEERING
# =========================================
cat_cols = ["AircraftModel", "PartName", "NatureOfConditionA", "StageOfOperationCode"]
num_cols = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
text_col = "Discrepancy"

df = engineer_features(df, cat_cols, text_col, date_col="DifficultyDate")

feature_cols = cat_cols + num_cols + [text_col]
X = df[feature_cols]
y = df["severity_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")


def build_preprocessor():
    """Fresh ColumnTransformer instance (sklearn transformers shouldn't be
    reused/re-fit across independent pipelines)."""
    return ColumnTransformer([
        ("text", TfidfVectorizer(max_features=400, stop_words="english", ngram_range=(1, 2)), text_col),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])


# =========================================
# 4. DEFINE THE 3 MODELS
# =========================================
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

models = {}

# --- Model 1: Logistic Regression ---
models["Logistic Regression"] = {
    "pipeline": Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
    "cv_X": X, "cv_y": y,
}

# --- Model 2: Random Forest ---
models["Random Forest"] = {
    "pipeline": Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_split=8, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        )),
    ]),
    "cv_X": X, "cv_y": y,
}

# --- Model 3: XGBoost (preferred) or HistGradientBoosting fallback ---
try:
    from xgboost import XGBClassifier 
    third_name = "XGBoost"
    third_pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    third_cv_X, third_cv_y = X, y
    print("\nUsing XGBoost as the 3rd model.")
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    third_name = "HistGradientBoosting (fallback, subsampled)"
    third_pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)),
        ("clf", HistGradientBoostingClassifier(max_iter=300, max_depth=8, random_state=RANDOM_STATE)),
    ])
    # HistGradientBoosting needs dense input; to keep memory/runtime sane on
    # the full ~300k-row dataset we fit/CV it on a stratified subsample.
    SUBSAMPLE_N = min(50000, len(X))
    if SUBSAMPLE_N >= len(X):
        third_cv_X, third_cv_y = X, y
    else:
        sub_idx, _ = train_test_split(
            np.arange(len(X)), train_size=SUBSAMPLE_N, stratify=y, random_state=RANDOM_STATE
        )
        third_cv_X, third_cv_y = X.iloc[sub_idx], y.iloc[sub_idx]
    print(f"\nxgboost not installed -> run `pip install xgboost` for the recommended 3rd model.")
    print(f"Falling back to sklearn HistGradientBoostingClassifier on a {SUBSAMPLE_N:,}-row stratified subsample.")

models[third_name] = {"pipeline": third_pipeline, "cv_X": third_cv_X, "cv_y": third_cv_y}

# =========================================
# 5. FIT + HOLDOUT EVALUATION
# =========================================
holdout_metrics = {}
roc_data = {}
pr_data = {}
fitted_pipelines = {}

for name, cfg in models.items():
    print(f"\n===== Fitting: {name} =====")
    pipe = cfg["pipeline"]
    # 3rd-model fallback trains on the subsample; LR/RF train on the full train split
    if name == third_name and third_name.startswith("HistGradientBoosting"):
        sub_n = min(40000, len(X_train))
        if sub_n >= len(X_train):
            pipe.fit(X_train, y_train)
        else:
            tr_idx, _ = train_test_split(
                np.arange(len(X_train)), train_size=sub_n,
                stratify=y_train, random_state=RANDOM_STATE
            )
            pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    else:
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    holdout_metrics[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc(fpr, tpr))
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_data[name] = (prec, rec)

    fitted_pipelines[name] = pipe

# =========================================
# 6. VISUALIZATIONS
# =========================================
print("\nGenerating visualizations...")

# --- Confusion matrices grid ---
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
if len(models) == 1:
    axes = [axes]
for ax, (name, pipe) in zip(axes, fitted_pipelines.items()):
    cm = confusion_matrix(y_test, pipe.predict(X_test))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Critical", "Critical"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(name)
fig.suptitle("Confusion Matrices - Holdout Test Set", y=1.03)
save_fig(fig, "02_confusion_matrices.png")

# --- ROC curves overlay ---
fig, ax = plt.subplots(figsize=(6, 6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
save_fig(fig, "03_roc_curves.png")

# --- Precision-Recall curves overlay ---
fig, ax = plt.subplots(figsize=(6, 6))
for name, (prec, rec) in pr_data.items():
    ax.plot(rec, prec, label=name)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve Comparison")
ax.legend()
save_fig(fig, "04_precision_recall_curves.png")

# --- Holdout F1 comparison bar ---
fig, ax = plt.subplots(figsize=(6, 4))
names = list(holdout_metrics.keys())
f1s = [holdout_metrics[n]["f1"] for n in names]
bars = ax.bar(names, f1s, color=sns.color_palette("Set2", len(names)))
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.set_title("Model Comparison - Holdout F1 Score")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
for b, v in zip(bars, f1s):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
save_fig(fig, "05_model_comparison_f1.png")

# --- Feature importance (tree models only) ---
for name in fitted_pipelines:
    clf = fitted_pipelines[name].named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        prep = fitted_pipelines[name].named_steps["prep"]
        feat_names = list(prep.named_transformers_["text"].get_feature_names_out()) + \
                     list(prep.named_transformers_["cat"].get_feature_names_out()) + num_cols
        importances = clf.feature_importances_
        feat_imp = pd.DataFrame({"feature": feat_names, "importance": importances}) \
            .sort_values("importance", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(feat_imp["feature"], feat_imp["importance"], color="#4C72B0")
        ax.invert_yaxis()
        ax.set_title(f"Top 20 Feature Importances - {name}")
        save_fig(fig, f"06_feature_importance_{name.split()[0].lower()}.png")

# =========================================
# 7. CROSS VALIDATION
# =========================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_f1_scores = {}       # section: 5-fold, single metric (F1)
cv_multi_results = {}   # section: 5-fold, multi-metric

for name, cfg in models.items():
    print(f"\n===== 5-Fold CV: {name} =====")
    pipe = cfg["pipeline"]
    cv_X, cv_y = cfg["cv_X"], cfg["cv_y"]

    # --- Single-metric 5-fold CV (F1) ---
    f1_scores = cross_val_score(pipe, cv_X, cv_y, cv=skf, scoring="f1", n_jobs=1)
    cv_f1_scores[name] = f1_scores
    print("F1 scores:", np.round(f1_scores, 3))
    print("Mean F1:", f1_scores.mean().round(3), "+/-", f1_scores.std().round(3))

    # --- Multi-metric 5-fold CV ---
    cv_results = cross_validate(pipe, cv_X, cv_y, cv=skf, scoring=scoring, n_jobs=1)
    cv_multi_results[name] = cv_results
    for m in scoring:
        print(f"{m.upper()}: {cv_results['test_' + m].mean():.3f} (+/- {cv_results['test_' + m].std():.3f})")

# --- CV F1 boxplot across models ---
fig, ax = plt.subplots(figsize=(6, 5))
try:
    ax.boxplot([cv_f1_scores[n] for n in models], tick_labels=list(models.keys()))
except TypeError:  # older matplotlib (<3.9) doesn't have tick_labels yet
    ax.boxplot([cv_f1_scores[n] for n in models], labels=list(models.keys()))
ax.set_ylabel("F1 Score")
ax.set_title("5-Fold CV F1 Score Distribution by Model")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
save_fig(fig, "07_cv_f1_boxplot.png")

# --- Multi-metric CV grouped bar chart ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(scoring))
width = 0.8 / len(models)
for i, name in enumerate(models):
    means = [cv_multi_results[name][f"test_{m}"].mean() for m in scoring]
    ax.bar(x + i * width, means, width, label=name)
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels([m.upper() for m in scoring])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("5-Fold Multi-Metric CV Comparison")
ax.legend()
save_fig(fig, "08_cv_multimetric_comparison.png")

# --- Prediction distribution for best model (chosen below) ---
best_model_name = max(models, key=lambda n: cv_multi_results[n]["test_f1"].mean())
best_pipe_holdout = fitted_pipelines[best_model_name]
y_pred_best = best_pipe_holdout.predict(X_test)

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=y_pred_best, ax=ax, palette=["#4C72B0", "#C44E52"])
ax.set_xticklabels(["Non-Critical (0)", "Critical (1)"])
ax.set_title(f"Prediction Distribution - {best_model_name} (Holdout)")
save_fig(fig, "09_prediction_distribution.png")

# =========================================
# 8. SELECT BEST MODEL, REFIT ON 100% DATA
# =========================================
print(f"\nBest model by mean CV F1: {best_model_name}")

final_pipeline = models[best_model_name]["pipeline"]
# Fresh unfitted clone so refit isn't polluted by CV's internal clones
if best_model_name == third_name and third_name.startswith("HistGradientBoosting"):
    # keep the fallback consistent: refit on the same subsample logic, but
    # using the FULL dataset's subsample rather than just the train split
    final_pipeline.fit(third_cv_X, third_cv_y)
    trained_rows_final = len(third_cv_X)
else:
    final_pipeline.fit(X, y)
    trained_rows_final = len(X)

joblib.dump(final_pipeline, os.path.join(MODEL_DIR, "severity_pipeline.joblib"))

# =========================================
# 9. SAVE METADATA + METRICS SUMMARY
# =========================================
metrics_rows = []
for name in models:
    row = {"model": name}
    row.update({f"holdout_{k}": v for k, v in holdout_metrics[name].items()})
    for m in scoring:
        row[f"cv_{m}_mean"] = cv_multi_results[name][f"test_{m}"].mean()
        row[f"cv_{m}_std"] = cv_multi_results[name][f"test_{m}"].std()
    metrics_rows.append(row)
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(os.path.join(MODEL_DIR, "model_comparison_metrics.csv"), index=False)
print("\nSaved metrics table -> models/model_comparison_metrics.csv")
print(metrics_df)

meta = {
    "best_model": best_model_name,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "text_col": text_col,
    "feature_cols": feature_cols,
    "trained_rows_final_fit": int(trained_rows_final),
    "trained_date_range": [str(df["DifficultyDate"].min()), str(df["DifficultyDate"].max())],
    "holdout_metrics": {k: {m: float(v) for m, v in vals.items()} for k, vals in holdout_metrics.items()},
    "cv_f1_mean": {n: float(cv_multi_results[n]["test_f1"].mean()) for n in models},
    "label_rule": {
        "critical_conditions": sorted(critical_conditions),
        "non_event_conditions": sorted(non_event_conditions),
        "text_fallback_keywords": TEXT_FALLBACK_KEYWORDS,
    },
}
with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved production pipeline -> models/severity_pipeline.joblib")
print("Saved metadata            -> models/feature_meta.json")
print("Saved plots                -> outputs/plots/*.png")
print("\nDone.")
