"""
Aircraft MRO Analytics Dashboard — REAL DATA VERSION
FAA SDR Dataset: Historical (2021-2025) + Recent (2026)

NOTE: This file was converted so the whole dashboard runs inside
run_dashboard() instead of at import time. Call run_dashboard() from
main_app.py. st.set_page_config() is NOT called here anymore — the
launcher (main_app.py) calls it once for the whole app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import json
import glob
warnings.filterwarnings("ignore")

# ── EDIT THESE TWO PATHS to point at your actual CSVs ──────────────────────
HIST_CSV = r"C:\Users\Priyanka\Downloads\faa_sdr_genai_project\genai_project\data\raw\Concated_Final_Cleaned_Dataset.csv"
REC_CSV  = r"C:\Users\Priyanka\Downloads\faa_sdr_genai_project\genai_project\data\raw\Cleaned_Output_FAA-SDR-2026.csv"

# ── Artifacts produced by 01_train_model.py / 02_predict_2026.py ───────────
# These are READ-ONLY here — the dashboard never retrains or re-scores
# anything, it only displays what those two scripts already computed and
# saved to disk. Run 01_train_model.py then 02_predict_2026.py first.
# Override with env vars if this app isn't run from the same folder as
# those scripts (they default to relative paths just like this).
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
PLOTS_DIR = os.environ.get("PLOTS_DIR", os.path.join("outputs", "plots"))
DASHBOARD_DATA_DIR = os.environ.get("DASHBOARD_DATA_DIR", os.path.join("outputs", "dashboard_data"))

@st.cache_data(show_spinner="Loading historical data (2021-2025)...")
def load_hist():
    df = pd.read_csv(HIST_CSV, index_col=0, low_memory=False)
    df["DifficultyDate"] = pd.to_datetime(df["DifficultyDate"], errors="coerce")
    df["SubmissionDate"]  = pd.to_datetime(df["SubmissionDate"],  errors="coerce")
    df["Year"]  = df["DifficultyDate"].dt.year
    df["Month"] = df["DifficultyDate"].dt.to_period("M").astype(str)
    return df

@st.cache_data(show_spinner="Loading 2026 data...")
def load_2026():
    df = pd.read_csv(REC_CSV, index_col=0, low_memory=False)
    df["DifficultyDate"] = pd.to_datetime(df["DifficultyDate"], errors="coerce")
    df["SubmissionDate"]  = pd.to_datetime(df["SubmissionDate"],  errors="coerce")
    df["Year"]  = df["DifficultyDate"].dt.year
    df["Month"] = df["DifficultyDate"].dt.to_period("M").astype(str)
    return df

# ── Loaders for 01_train_model.py / 02_predict_2026.py outputs ─────────────
@st.cache_data(show_spinner=False)
def load_training_artifacts():
    """Reads models/feature_meta.json + models/model_comparison_metrics.csv
    saved by 01_train_model.py. Returns (None, None) if not found — caller
    must show a clear error rather than fabricate numbers."""
    meta_path = os.path.join(MODEL_DIR, "feature_meta.json")
    metrics_path = os.path.join(MODEL_DIR, "model_comparison_metrics.csv")
    if not (os.path.exists(meta_path) and os.path.exists(metrics_path)):
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    metrics_df = pd.read_csv(metrics_path)
    return meta, metrics_df


@st.cache_data(show_spinner=False)
def load_prediction_artifacts():
    """Reads everything 02_predict_2026.py wrote to outputs/dashboard_data/.
    Returns None if the folder/KPI file is missing."""
    kpi_path = os.path.join(DASHBOARD_DATA_DIR, "summary_kpis.json")
    if not os.path.exists(kpi_path):
        return None
    with open(kpi_path) as f:
        kpis = json.load(f)

    def _read(name):
        p = os.path.join(DASHBOARD_DATA_DIR, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    return {
        "kpis": kpis,
        "monthly": _read("monthly_trend.csv"),
        "by_part": _read("by_part_name.csv"),
        "by_model": _read("by_aircraft_model.csv"),
        "by_region": _read("by_region.csv"),
        "top_examples": _read("top_critical_examples.csv"),
        "detail": _read("predictions_detail.csv"),
    }


def artifact_img(filename, caption=None):
    """Displays a plot exactly as saved by the training/predict script.
    Never regenerates — shows a clear message if the file isn't there."""
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        st.image(path, use_container_width=True, caption=caption)
    else:
        ph(f"'{filename}' not found in {PLOTS_DIR}/ — run the training/predict script to generate it.")

PT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
          margin=dict(l=20,r=20,t=40,b=20))

def kpi_card(label,value,delta="",direction="neu"):
    dc={"up":"kpi-up","down":"kpi-down","neu":"kpi-neu"}.get(direction,"kpi-neu")
    dh=f'<div class="kpi-delta {dc}">{delta}</div>' if delta else ""
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{dh}</div>'

def sh(text): st.markdown(f'<div class="section-header">{text}</div>',unsafe_allow_html=True)
def ph(msg): st.markdown(f'<div class="placeholder">📊 {msg}</div>',unsafe_allow_html=True)
def ib(msg): st.markdown(f'<div class="info-banner">ℹ️ {msg}</div>',unsafe_allow_html=True)
def top_n(s,n=10): return s.value_counts().head(n).reset_index().set_axis([s.name,"Count"],axis=1)

def compute_kpis(df,label):
    total=len(df)
    makes=df["AircraftMake"].nunique() if "AircraftMake" in df.columns else "N/A"
    parts=df["PartName"].nunique() if "PartName" in df.columns else "N/A"
    regions=df["ReceivingRegionCode"].nunique() if "ReceivingRegionCode" in df.columns else "N/A"
    if "DifficultyDate" in df.columns and "SubmissionDate" in df.columns:
        sub = pd.to_datetime(df["SubmissionDate"], errors="coerce", utc=True)
        diff = pd.to_datetime(df["DifficultyDate"], errors="coerce", utc=True)
        days = (sub - diff).dt.days
        avg_d=f"{days.dropna().mean():.1f}D"
    else: avg_d="N/A"
    return {
        "total":   {"value":f"{total:,}","delta":label,"direction":"neu"},
        "mttr":    {"value":avg_d,"delta":"Avg days to submit","direction":"neu"},
        "makes":   {"value":str(makes),"delta":"Aircraft makes","direction":"neu"},
        "parts":   {"value":str(parts),"delta":"Unique parts","direction":"neu"},
        "regions": {"value":str(regions),"delta":"FAA regions","direction":"neu"},
        "years":   {"value":str(df["Year"].nunique()) if "Year" in df.columns else "N/A","delta":"Years covered","direction":"neu"},
    }

def render_kpi_row(kpis):
    defs=[("total","Total SDRs"),("mttr","Days to Submit"),("makes","Aircraft Makes"),
          ("parts","Unique Parts"),("regions","FAA Regions"),("years","Years")]
    cols=st.columns(6)
    for col,(key,lbl) in zip(cols,defs):
        item=kpis.get(key,{"value":"--","delta":"","direction":"neu"})
        col.markdown(kpi_card(lbl,item["value"],item.get("delta",""),item.get("direction","neu")),unsafe_allow_html=True)

def render_eda_tab(df,label):
    ib(f"EDA — {label}  |  {len(df):,} SDR records")
    sh("Top 15 Aircraft Makes by SDR Count")
    c1,c2=st.columns([3,2])
    with c1:
        mk=top_n(df["AircraftMake"],15)
        fig=px.bar(mk.sort_values("Count"),x="Count",y="AircraftMake",orientation="h",
                   color="Count",color_continuous_scale="Blues",title="SDR Reports by Aircraft Make")
        fig.update_layout(**PT,height=450,coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig2=px.pie(mk.head(8),names="AircraftMake",values="Count",title="Top 8 Makes",
                    hole=0.45,color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_layout(**PT,height=450)
        st.plotly_chart(fig2,use_container_width=True)
    sh("Part Condition & SDR Type")
    c3,c4=st.columns(2)
    with c3:
        pc=top_n(df["PartCondition"],10)
        fig3=px.bar(pc,x="Count",y="PartCondition",orientation="h",
                    color="Count",color_continuous_scale="Oranges",title="Part Condition Codes")
        fig3.update_layout(**PT,height=380,coloraxis_showscale=False)
        st.plotly_chart(fig3,use_container_width=True)
    with c4:
        sdr=top_n(df["SDRType"],8)
        fig4=px.pie(sdr,names="SDRType",values="Count",title="SDR Type Breakdown",hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Teal)
        fig4.update_layout(**PT,height=380)
        st.plotly_chart(fig4,use_container_width=True)
    sh("Aircraft Total Time & Cycles at Difficulty")
    c5,c6=st.columns(2)
    with c5:
        att=df["AircraftTotalTime"].dropna()
        att=att[(att>0)&(att<att.quantile(0.99))]
        fig5=px.histogram(att,nbins=50,color_discrete_sequence=["#3b82f6"],
                          title="Aircraft Total Time (hrs)",labels={"value":"Total Time (hrs)"})
        fig5.update_layout(**PT,height=350)
        st.plotly_chart(fig5,use_container_width=True)
    with c6:
        atc=df["AircraftTotalCycles"].dropna()
        atc=atc[(atc>0)&(atc<atc.quantile(0.99))]
        fig6=px.histogram(atc,nbins=50,color_discrete_sequence=["#f59e0b"],
                          title="Aircraft Total Cycles",labels={"value":"Total Cycles"})
        fig6.update_layout(**PT,height=350)
        st.plotly_chart(fig6,use_container_width=True)
    sh("How Was the Difficulty Discovered?")
    hd=top_n(df["HowDiscoveredCode"],12)
    fig7=px.bar(hd,x="HowDiscoveredCode",y="Count",color="Count",
                color_continuous_scale="Blues",title="Discovery Method Codes")
    fig7.update_layout(**PT,height=350,coloraxis_showscale=False)
    st.plotly_chart(fig7,use_container_width=True)
    sh("Top 15 Parts with Most SDR Reports")
    pts=top_n(df["PartName"],15)
    fig8=px.bar(pts.sort_values("Count"),x="Count",y="PartName",orientation="h",
                color="Count",color_continuous_scale="Reds",title="Parts by SDR Frequency")
    fig8.update_layout(**PT,height=480,coloraxis_showscale=False)
    st.plotly_chart(fig8,use_container_width=True)
    with st.expander("Raw Data Preview (500 rows)"):
        st.dataframe(df.head(500),use_container_width=True)

def render_trend_tab(df,label):
    ib(f"Trend Analysis — {label}")
    sh("Monthly SDR Report Trend")
    monthly=df.groupby(df["DifficultyDate"].dt.to_period("M")).size().reset_index()
    monthly.columns=["Period","Count"]
    monthly["Date"]=monthly["Period"].astype(str)
    monthly["Rolling3M"]=monthly["Count"].rolling(3,min_periods=1).mean()
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=monthly["Date"],y=monthly["Count"],name="Monthly SDRs",
                         marker_color="#21262d"),secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["Date"],y=monthly["Rolling3M"],name="3M Avg",
                             line=dict(color="#ef4444",width=2.5)),secondary_y=True)
    fig.update_yaxes(title_text="SDR Count",secondary_y=False,color="#3b82f6")
    fig.update_yaxes(title_text="Rolling Avg",secondary_y=True,color="#ef4444")
    fig.update_layout(title=f"Monthly SDR Volume — {label}",height=420,**PT)
    st.plotly_chart(fig,use_container_width=True)
    sh("Year-over-Year SDR Volume")
    yoy=df.groupby("Year").size().reset_index(name="Count")
    fig2=px.bar(yoy,x="Year",y="Count",color="Count",color_continuous_scale="Blues",
                title="SDR Reports per Year")
    fig2.update_layout(**PT,height=360,coloraxis_showscale=False)
    st.plotly_chart(fig2,use_container_width=True)
    sh("SDR Volume by FAA Region Over Time")
    top_reg=df["ReceivingRegionCode"].value_counts().head(6).index
    reg_df=df[df["ReceivingRegionCode"].isin(top_reg)].copy()
    reg_m=reg_df.groupby([reg_df["DifficultyDate"].dt.to_period("M").astype(str),
                          "ReceivingRegionCode"]).size().reset_index(name="Count")
    reg_m.columns=["Month","Region","Count"]
    fig3=px.line(reg_m,x="Month",y="Count",color="Region",title="Top 6 Regions — Monthly Trend")
    fig3.update_layout(**PT,height=420)
    st.plotly_chart(fig3,use_container_width=True)
    sh("Top 5 Aircraft Makes — Trend Over Time")
    top_mk=df["AircraftMake"].value_counts().head(5).index
    mk_df=df[df["AircraftMake"].isin(top_mk)].copy()
    mk_m=mk_df.groupby([mk_df["DifficultyDate"].dt.to_period("M").astype(str),
                        "AircraftMake"]).size().reset_index(name="Count")
    mk_m.columns=["Month","Make","Count"]
    fig4=px.line(mk_m,x="Month",y="Count",color="Make",title="Top 5 Makes — Monthly Volume")
    fig4.update_layout(**PT,height=420)
    st.plotly_chart(fig4,use_container_width=True)
    sh("Part Condition x Region Heatmap")
    top_pc=df["PartCondition"].value_counts().head(8).index
    top_rg=df["ReceivingRegionCode"].value_counts().head(8).index
    hdf=df[df["PartCondition"].isin(top_pc)&df["ReceivingRegionCode"].isin(top_rg)]
    pivot=hdf.pivot_table(index="PartCondition",columns="ReceivingRegionCode",aggfunc="size",fill_value=0)
    fig5=px.imshow(pivot,color_continuous_scale="Blues",aspect="auto",
                   title="SDR Density: Part Condition x FAA Region")
    fig5.update_layout(**PT,height=400)
    st.plotly_chart(fig5,use_container_width=True)

def render_nlp_tab(df,label):
    ib(f"NLP & Root Cause Analysis — {label}")
    sh("Top Terms in Discrepancy Field")
    from collections import Counter; import re
    sw={"the","a","an","and","or","of","to","in","is","was","were","for","on","at",
        "by","with","from","that","this","it","as","be","has","had","have","are","not","no"}
    text=df["Discrepancy"].dropna().astype(str).str.lower().str.cat(sep=" ")
    words=re.findall(r'\b[a-z]{3,}\b',text)
    wf=Counter(w for w in words if w not in sw)
    wf_df=pd.DataFrame(wf.most_common(20),columns=["Term","Frequency"])
    fig=px.bar(wf_df.sort_values("Frequency"),x="Frequency",y="Term",orientation="h",
               color="Frequency",color_continuous_scale="Oranges",title="Top 20 Discrepancy Terms")
    fig.update_layout(**PT,height=500,coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)
    sh("Root Cause Pareto — Nature of Condition A")
    noc=df["NatureOfConditionA"].value_counts().head(15).reset_index()
    noc.columns=["Condition","Count"]
    cumulative=noc["Count"].cumsum()/noc["Count"].sum()*100
    fig2=make_subplots(specs=[[{"secondary_y":True}]])
    fig2.add_trace(go.Bar(x=noc["Condition"],y=noc["Count"],name="Count",
                          marker_color="#3b82f6"),secondary_y=False)
    fig2.add_trace(go.Scatter(x=noc["Condition"],y=cumulative,name="Cumulative %",
                              mode="lines+markers",line=dict(color="#f59e0b",width=2.5)),secondary_y=True)
    fig2.add_hline(y=80,line_dash="dash",line_color="#ef4444",secondary_y=True,
                   annotation_text="80%",annotation_position="top right")
    fig2.update_yaxes(title_text="SDR Count",secondary_y=False)
    fig2.update_yaxes(title_text="Cumulative %",range=[0,105],secondary_y=True)
    fig2.update_layout(title="Pareto — Nature of Condition A",height=450,**PT)
    st.plotly_chart(fig2,use_container_width=True)
    sh("How Discovered x Part Condition Heatmap")
    top_hd=df["HowDiscoveredCode"].value_counts().head(8).index
    top_pc=df["PartCondition"].value_counts().head(6).index
    cross=df[df["HowDiscoveredCode"].isin(top_hd)&df["PartCondition"].isin(top_pc)]
    cp=cross.pivot_table(index="HowDiscoveredCode",columns="PartCondition",aggfunc="size",fill_value=0)
    fig3=px.imshow(cp,color_continuous_scale="Blues",aspect="auto",
                   title="Discovery Method x Part Condition")
    fig3.update_layout(**PT,height=400)
    st.plotly_chart(fig3,use_container_width=True)
    sh("SDR by Stage of Operation")
    soo=df["StageOfOperationCode"].value_counts().head(12).reset_index()
    soo.columns=["Stage","Count"]
    fig4=px.bar(soo,x="Stage",y="Count",color="Count",color_continuous_scale="Teal",
                title="Failures by Stage of Operation")
    fig4.update_layout(**PT,height=380,coloraxis_showscale=False)
    st.plotly_chart(fig4,use_container_width=True)

def render_training_results_tab():
    """Displays results saved by 01_train_model.py — read-only, no retraining
    or re-evaluation happens here. If this errors, run that script first."""
    ib("Model Training Results — from 01_train_model.py (2021-2025 historical data)")

    meta, metrics_df = load_training_artifacts()
    if meta is None:
        st.error(
            f"No training artifacts found in `{MODEL_DIR}/`. "
            "Run 01_train_model.py first, then reload this page."
        )
        return

    best_model = meta["best_model"]
    sh("Selected Model & Cross-Validated Performance")
    st.markdown(
        f"**Best model (chosen by 5-fold CV F1):** `{best_model}`  \n"
        f"Trained on **{meta['trained_rows_final_fit']:,}** rows "
        f"({meta['trained_date_range'][0][:10]} to {meta['trained_date_range'][1][:10]})"
    )

    best_row = metrics_df.loc[metrics_df["model"] == best_model].iloc[0]
    mcols = st.columns(5)
    for col, (key, lbl) in zip(mcols, [
        ("cv_accuracy_mean", "Accuracy"), ("cv_precision_mean", "Precision"),
        ("cv_recall_mean", "Recall"), ("cv_f1_mean", "F1 Score"),
        ("cv_roc_auc_mean", "ROC-AUC"),
    ]):
        val = float(best_row[key])
        color = "#22c55e" if val >= .90 else "#f59e0b" if val >= .80 else "#ef4444"
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{lbl} (5-fold CV)</div>'
            f'<div class="kpi-value" style="color:{color}">{val:.3f}</div></div>',
            unsafe_allow_html=True)
    st.caption("These are cross-validated metrics from training — not recomputed live, "
               "and not scored on data the final model was refit on.")

    sh("Full Model Comparison — All 3 Algorithms")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    sh("Severity Label Distribution (2021-2025)")
    artifact_img("01_severity_label_distribution.png")

    sh("Confusion Matrices — Holdout Test Set")
    artifact_img("02_confusion_matrices.png")

    c1, c2 = st.columns(2)
    with c1:
        sh("ROC Curve Comparison")
        artifact_img("03_roc_curves.png")
    with c2:
        sh("Precision-Recall Curve Comparison")
        artifact_img("04_precision_recall_curves.png")

    sh("Holdout F1 Score Comparison")
    artifact_img("05_model_comparison_f1.png")

    sh("Feature Importance")
    fi_files = sorted(glob.glob(os.path.join(PLOTS_DIR, "06_feature_importance_*.png")))
    if fi_files:
        fi_cols = st.columns(len(fi_files))
        for col, fpath in zip(fi_cols, fi_files):
            with col:
                st.image(fpath, use_container_width=True)
    else:
        ph("No feature importance plots found (only tree-based models produce these).")

    sh("5-Fold Cross-Validation")
    c3, c4 = st.columns(2)
    with c3:
        artifact_img("07_cv_f1_boxplot.png")
    with c4:
        artifact_img("08_cv_multimetric_comparison.png")

    sh("Prediction Distribution (Holdout, Best Model)")
    artifact_img("09_prediction_distribution.png")

    with st.expander("Label rule reference — how severity_flag was assigned"):
        st.json(meta["label_rule"])


def render_prediction_results_tab():
    """Displays results saved by 02_predict_2026.py — read-only. The model
    is never re-run here; every number/plot below was computed once by that
    script and simply loaded from disk."""
    ib("2026 Predictions — from 02_predict_2026.py (unseen data, scored by the saved model)")

    data = load_prediction_artifacts()
    if data is None:
        st.error(
            f"No prediction artifacts found in `{DASHBOARD_DATA_DIR}/`. "
            "Run 02_predict_2026.py first, then reload this page."
        )
        return

    kpis = data["kpis"]
    sh("Headline KPIs")
    st.caption(f"Model used: `{kpis['model_used']}`  |  "
               f"trained through {kpis['model_trained_through'][:10]}  |  "
               f"scoring {kpis['date_range'][0][:10]} to {kpis['date_range'][1][:10]}")
    kcols = st.columns(5)
    kdefs = [
        ("total_records_2026", "Total 2026 Records", "{:,}"),
        ("predicted_critical_count", "Predicted Critical", "{:,}"),
        ("predicted_non_critical_count", "Predicted Non-Critical", "{:,}"),
        ("predicted_critical_rate", "Critical Rate", "{:.1%}"),
        ("avg_critical_probability", "Avg. Critical Probability", "{:.3f}"),
    ]
    for col, (key, lbl, fmt) in zip(kcols, kdefs):
        col.markdown(kpi_card(lbl, fmt.format(kpis[key])), unsafe_allow_html=True)

    sh("Monthly Volume & Predicted Critical Rate")
    artifact_img("01_2026_monthly_trend.png")
    if not data["monthly"].empty:
        st.dataframe(data["monthly"], use_container_width=True, hide_index=True)

    sh("Top Parts & Aircraft Models by Predicted Critical Volume")
    c1, c2 = st.columns(2)
    with c1:
        artifact_img("02_2026_top_parts_critical.png")
        if not data["by_part"].empty:
            st.dataframe(data["by_part"].head(15), use_container_width=True, hide_index=True)
    with c2:
        artifact_img("03_2026_top_models_critical.png")
        if not data["by_model"].empty:
            st.dataframe(data["by_model"].head(15), use_container_width=True, hide_index=True)

    if not data["by_region"].empty:
        sh("Regional Breakdown")
        st.dataframe(data["by_region"], use_container_width=True, hide_index=True)

    sh("Prediction & Probability Distribution")
    c3, c4 = st.columns(2)
    with c3:
        artifact_img("04_2026_prediction_distribution.png")
    with c4:
        artifact_img("05_2026_probability_histogram.png")

    if not data["top_examples"].empty:
        sh("Top 100 Highest-Risk Records — Watchlist")
        st.dataframe(data["top_examples"], use_container_width=True, hide_index=True)

    if not data["detail"].empty:
        with st.expander(f"Full row-level predictions ({len(data['detail']):,} records)"):
            st.dataframe(data["detail"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download predictions_detail.csv",
                data["detail"].to_csv(index=False),
                "predictions_detail.csv", "text/csv"
            )


def render_comparative_page(df_hist, df_2026):
    st.markdown("## Comparative Insights — 2021-2025 vs 2026")
    ib("Live comparison using your real FAA SDR datasets.")
    sh("Key Metrics — Head to Head")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### Historical (2021–2025)")
        st.metric("Total SDR Reports",f"{len(df_hist):,}")
        st.metric("Unique Aircraft Makes",df_hist["AircraftMake"].nunique())
        st.metric("Unique Parts",df_hist["PartName"].nunique())
        st.metric("FAA Regions",df_hist["ReceivingRegionCode"].nunique())
    with c2:
        st.markdown("### Recent (2026)")
        st.metric("Total SDR Reports",f"{len(df_2026):,}")
        st.metric("Unique Aircraft Makes",df_2026["AircraftMake"].nunique())
        st.metric("Unique Parts",df_2026["PartName"].nunique())
        st.metric("FAA Regions",df_2026["ReceivingRegionCode"].nunique())
    sh("Monthly SDR Volume — Overlaid")
    hm=df_hist.groupby(df_hist["DifficultyDate"].dt.to_period("M")).size().reset_index()
    hm.columns=["Period","Count"]; hm["Date"]=hm["Period"].astype(str)
    rm=df_2026.groupby(df_2026["DifficultyDate"].dt.to_period("M")).size().reset_index()
    rm.columns=["Period","Count"]; rm["Date"]=rm["Period"].astype(str)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hm["Date"],y=hm["Count"],name="2021-2025",
                             line=dict(color="#3b82f6",width=2)))
    fig.add_trace(go.Scatter(x=rm["Date"],y=rm["Count"],name="2026",
                             line=dict(color="#f59e0b",width=2.5)))
    fig.update_layout(title="Monthly SDR Count Comparison",height=420,**PT)
    st.plotly_chart(fig,use_container_width=True)
    sh("Top 10 Aircraft Makes — Historical vs 2026")
    th=df_hist["AircraftMake"].value_counts().head(10).reset_index()
    th.columns=["Make","Count"]; th["Period"]="2021-2025"
    tr=df_2026["AircraftMake"].value_counts().head(10).reset_index()
    tr.columns=["Make","Count"]; tr["Period"]="2026"
    fig2=px.bar(pd.concat([th,tr]),x="Make",y="Count",color="Period",barmode="group",
                color_discrete_map={"2021-2025":"#3b82f6","2026":"#f59e0b"},title="Aircraft Make Comparison")
    fig2.update_layout(**PT,height=420)
    st.plotly_chart(fig2,use_container_width=True)
    sh("Part Condition — Historical vs 2026")
    ph_=df_hist["PartCondition"].value_counts().head(10).reset_index()
    ph_.columns=["Condition","Count"]; ph_["Period"]="2021-2025"
    pr=df_2026["PartCondition"].value_counts().head(10).reset_index()
    pr.columns=["Condition","Count"]; pr["Period"]="2026"
    fig3=px.bar(pd.concat([ph_,pr]),x="Condition",y="Count",color="Period",barmode="group",
                color_discrete_map={"2021-2025":"#3b82f6","2026":"#f59e0b"},title="Part Condition Comparison")
    fig3.update_layout(**PT,height=420)
    st.plotly_chart(fig3,use_container_width=True)
    sh("Nature of Condition A — Historical vs 2026")
    nh=df_hist["NatureOfConditionA"].value_counts().head(10).reset_index()
    nh.columns=["Condition","Count"]; nh["Period"]="2021-2025"
    nr=df_2026["NatureOfConditionA"].value_counts().head(10).reset_index()
    nr.columns=["Condition","Count"]; nr["Period"]="2026"
    fig4=px.bar(pd.concat([nh,nr]),x="Condition",y="Count",color="Period",barmode="group",
                color_discrete_map={"2021-2025":"#3b82f6","2026":"#f59e0b"},title="Condition Code Comparison")
    fig4.update_layout(**PT,height=420)
    st.plotly_chart(fig4,use_container_width=True)


def render_early_warning_tab(df, label):
    ib(f"Early Warning System — {label}")
    st.caption("This tab uses an independent keyword-based **Heuristic Risk Score** (0-3) "
               "for fast, rule-based screening — it is separate from the trained ML "
               "classifier's `severity_flag` shown in the Predictive Maintenance tab.")

    df = df.copy()

    severity_score_map = {
        "failure":3,"fail":3,"crack":3,"broken":3,
        "leak":2,"pressure":2,"corrosion":2,"overheat":2,
        "seat":1,"door":1,
    }

    def get_severity_score(text):
        text = str(text).lower()
        for word, score in severity_score_map.items():
            if word in text:
                return score
        return 0

    if "SeverityScore" not in df.columns:
        df["SeverityScore"] = df["Discrepancy"].apply(get_severity_score)

    alerts = pd.Series(dtype=float)  # default so later refs don't NameError

    sh("🔴 Step 1 — Top High-Risk Parts (Latest Year)")
    if "Year" in df.columns and "PartName" in df.columns:
        latest_year = int(df["Year"].max())
        recent_data = df[df["Year"] == latest_year]

        st.markdown(f"**Analysing latest year: {latest_year} "
                    f"({len(recent_data):,} records)**")

        part_risk = (recent_data
                     .groupby("PartName")["SeverityScore"]
                     .mean()
                     .sort_values(ascending=False)
                     .head(15)
                     .reset_index())
        part_risk.columns = ["Part Name", "Avg Severity Score"]

        fig = px.bar(part_risk.sort_values("Avg Severity Score"),
                     x="Avg Severity Score", y="Part Name",
                     orientation="h",
                     color="Avg Severity Score",
                     color_continuous_scale="Reds",
                     title=f"Top 15 High-Risk Parts — {latest_year}")
        fig.update_layout(**PT, height=500, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        sh("⚠️ Step 2 — HIGH-RISK ALERT PARTS (Severity > 2.5)")
        THRESHOLD = 2.5

        part_risk_full = (recent_data
                          .groupby("PartName")["SeverityScore"]
                          .mean()
                          .sort_values(ascending=False))
        alerts = part_risk_full[part_risk_full > THRESHOLD]

        if len(alerts) > 0:
            alert_df = alerts.reset_index()
            alert_df.columns = ["Part Name", "Avg Severity Score"]
            alert_df["Risk Level"] = alert_df["Avg Severity Score"].apply(
                lambda x: "🔴 CRITICAL" if x >= 3 else "🟠 HIGH"
            )
            alert_df["Action Required"] = "Immediate maintenance attention required"

            st.markdown(
                f'<div style="background:rgba(239,68,68,0.15);border:1px solid #ef4444;'
                f'border-radius:8px;padding:1rem;margin-bottom:1rem;">'
                f'<b style="color:#ef4444;">⚠️ {len(alerts)} PARTS REQUIRE '
                f'IMMEDIATE ATTENTION</b><br>'
                f'<span style="color:#e6edf3;font-size:.85rem;">'
                f'These parts have average severity score above {THRESHOLD} '
                f'— immediate maintenance review recommended.</span></div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                alert_df.style.background_gradient(
                    subset=["Avg Severity Score"], cmap="Reds"
                ),
                use_container_width=True,
                hide_index=True
            )

            fig2 = px.bar(alert_df.sort_values("Avg Severity Score"),
                          x="Avg Severity Score", y="Part Name",
                          orientation="h",
                          color="Avg Severity Score",
                          color_continuous_scale="Reds",
                          title=f"⚠️ Alert Parts — Severity > {THRESHOLD}")
            fig2.add_vline(x=THRESHOLD, line_dash="dash",
                           line_color="#f59e0b",
                           annotation_text=f"Threshold ({THRESHOLD})",
                           annotation_position="top right")
            fig2.update_layout(**PT, height=400, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.success("✅ No critical alerts — all parts below severity threshold.")
    else:
        THRESHOLD = 2.5

    sh("📈 Step 3 — Severity Trend Over Years")
    if "Year" in df.columns:
        trend = (df.groupby("Year")["SeverityScore"]
                 .mean()
                 .reset_index())
        trend.columns = ["Year", "Avg Severity Score"]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=trend["Year"], y=trend["Avg Severity Score"],
            mode="lines+markers",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=10, color="#ef4444",
                        line=dict(width=2, color="#0d1117")),
            name="Avg Severity Score",
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.1)"
        ))
        fig3.add_hline(y=THRESHOLD, line_dash="dash",
                       line_color="#f59e0b",
                       annotation_text=f"Alert Threshold ({THRESHOLD})",
                       annotation_position="top right")
        fig3.update_layout(
            title="Average Severity Score Over Years",
            xaxis_title="Year",
            yaxis_title="Average Severity Score",
            height=400, **PT
        )
        st.plotly_chart(fig3, use_container_width=True)

        if len(trend) >= 2:
            first_val = trend["Avg Severity Score"].iloc[0]
            last_val  = trend["Avg Severity Score"].iloc[-1]
            change    = last_val - first_val
            if change > 0.1:
                st.markdown(
                    '<div style="background:rgba(239,68,68,0.1);border-left:'
                    '3px solid #ef4444;padding:.7rem 1rem;border-radius:4px;">'
                    '⬆️ <b>Increasing trend</b> — system degradation detected. '
                    'Escalate maintenance frequency.</div>',
                    unsafe_allow_html=True)
            elif change < -0.1:
                st.markdown(
                    '<div style="background:rgba(34,197,94,0.1);border-left:'
                    '3px solid #22c55e;padding:.7rem 1rem;border-radius:4px;">'
                    '⬇️ <b>Decreasing trend</b> — maintenance program is effective. '
                    'Continue current strategy.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="background:rgba(245,158,11,0.1);border-left:'
                    '3px solid #f59e0b;padding:.7rem 1rem;border-radius:4px;">'
                    '➡️ <b>Stable trend</b> — severity score is consistent. '
                    'Monitor for changes.</div>',
                    unsafe_allow_html=True)

    sh("📋 Step 4 — Early Warning Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_card(
            "High-Risk Parts",
            str(len(alerts)),
            "Parts above severity 2.5",
            "down" if len(alerts) > 0 else "up"
        ), unsafe_allow_html=True)
    with c2:
        avg_sev = df["SeverityScore"].mean()
        st.markdown(kpi_card(
            "Fleet Avg Severity",
            f"{avg_sev:.2f}",
            "⬆️ Escalate" if avg_sev > 2 else "✅ Nominal",
            "down" if avg_sev > 2 else "up"
        ), unsafe_allow_html=True)
    with c3:
        high_risk_count = len(df[df["SeverityScore"] >= 3])
        st.markdown(kpi_card(
            "Critical Events",
            f"{high_risk_count:,}",
            "Severity score = 3",
            "down" if high_risk_count > 0 else "up"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:var(--surface);border:1px solid var(--border);
    border-radius:8px;padding:1.2rem;">
    <b style="color:#3b82f6;">Early Warning System — Key Takeaways</b><br><br>
    🔴 <b>High severity parts</b> indicate potential failure risk<br>
    📈 <b>Increasing trend</b> indicates system degradation<br>
    ⚠️ <b>Alerts</b> help detect issues before they become critical<br>
    ✅ <b>Parts below threshold</b> are within acceptable maintenance limits
    </div>
    """, unsafe_allow_html=True)


def run_dashboard():
    """Entry point — call this from main_app.py to render the whole
    dashboard. Does NOT call st.set_page_config (the launcher does that
    once for the whole multi-app)."""

    # Dashboard-specific CSS, scoped to this call
    st.markdown("""<style>
    :root{--bg:#0d1117;--surface:#161b22;--border:#21262d;--accent:#3b82f6;
    --accent2:#f59e0b;--danger:#ef4444;--success:#22c55e;--text:#e6edf3;--muted:#8b949e;}
    html,body,[class*="css"]{background-color:var(--bg);color:var(--text);font-family:'Inter','Segoe UI',sans-serif;}
    .stApp{background-color:var(--bg);}
    [data-testid="stSidebar"]{background-color:var(--surface);border-right:1px solid var(--border);}
    .kpi-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
    padding:1.1rem 1.2rem;text-align:center;transition:border-color .2s;}
    .kpi-card:hover{border-color:var(--accent);}
    .kpi-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.35rem;}
    .kpi-value{font-size:1.75rem;font-weight:700;color:var(--text);line-height:1.1;}
    .kpi-delta{font-size:.78rem;margin-top:.25rem;}
    .kpi-up{color:var(--success);}.kpi-down{color:var(--danger);}.kpi-neu{color:var(--muted);}
    .section-header{font-size:1.05rem;font-weight:600;color:var(--accent);text-transform:uppercase;
    letter-spacing:.06em;border-bottom:1px solid var(--border);padding-bottom:.4rem;margin:1.4rem 0 .8rem;}
    [data-testid="stTabs"] [role="tab"]{color:var(--muted);font-size:.85rem;font-weight:500;}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--accent);border-bottom:2px solid var(--accent);}
    .placeholder{background:var(--surface);border:1px dashed var(--border);border-radius:8px;
    padding:2rem;text-align:center;color:var(--muted);font-size:.85rem;}
    .info-banner{background:rgba(59,130,246,.1);border-left:3px solid var(--accent);border-radius:4px;
    padding:.7rem 1rem;font-size:.83rem;color:var(--text);margin-bottom:1rem;}
    footer{visibility:hidden;}
    </style>""", unsafe_allow_html=True)

    df_hist = load_hist()
    df_2026 = load_2026()

    with st.sidebar:
        st.markdown("## ✈️ MRO Analytics")
        st.markdown('<p style="color:#8b949e;font-size:.78rem;">FAA SDR Predictive Maintenance</p>',unsafe_allow_html=True)
        st.divider()
        view=st.radio("Navigation",options=[
            "Historical Analysis (2021-2025)",
            "Recent Analysis (2026)",
            "Comparative Insights",
        ],label_visibility="collapsed")
        st.divider()
        st.markdown(f'<p style="color:#8b949e;font-size:.75rem;">Historical records loaded</p>'
                    f'<p style="color:#e6edf3;font-weight:600;">{len(df_hist):,}</p>',unsafe_allow_html=True)
        st.markdown(f'<p style="color:#8b949e;font-size:.75rem;">2026 records loaded</p>'
                    f'<p style="color:#e6edf3;font-weight:600;">{len(df_2026):,}</p>',unsafe_allow_html=True)
        st.divider()
        with st.expander("🔧 Artifact status (debug)"):
            _meta, _metrics = load_training_artifacts()
            _pred = load_prediction_artifacts()
            st.markdown(f"`{MODEL_DIR}/` — {'✅ found' if _meta is not None else '❌ missing'}")
            st.markdown(f"`{PLOTS_DIR}/` — {'✅ found' if os.path.isdir(PLOTS_DIR) else '❌ missing'}")
            st.markdown(f"`{DASHBOARD_DATA_DIR}/` — {'✅ found' if _pred is not None else '❌ missing'}")
            st.caption(f"Resolved relative to: `{os.getcwd()}`")
            if _meta is None or _pred is None:
                st.caption("Run 01_train_model.py and 02_predict_2026.py from this same "
                           "folder (or set MODEL_DIR/PLOTS_DIR/DASHBOARD_DATA_DIR env vars) "
                           "before launching the dashboard.")
        st.divider()

    if "Historical" in view:
        st.markdown("## Historical MRO Analysis")
        render_kpi_row(compute_kpis(df_hist,"2021-2025"))
        st.markdown("<br>",unsafe_allow_html=True)
        t1,t2,t3,t4,t5=st.tabs(["📊 EDA & Visualisations","📈 Trend & Network Analysis",
                               "💬 NLP & Root Cause Analysis","🤖 Predictive Maintenance", "⚠️Early Warning System"])
        with t1: render_eda_tab(df_hist,"2021-2025")
        with t2: render_trend_tab(df_hist,"2021-2025")
        with t3: render_nlp_tab(df_hist,"2021-2025")
        with t4: render_training_results_tab()
        with t5: render_early_warning_tab(df_hist, "2021-2025")
    elif "Recent" in view:
        st.markdown("## Recent MRO Analysis")
        render_kpi_row(compute_kpis(df_2026,"2026 YTD"))
        st.markdown("<br>",unsafe_allow_html=True)
        t1,t2,t3,t4,t5=st.tabs(["📊 EDA & Visualisations","📈 Trend & Network Analysis",
                               "💬 NLP & Root Cause Analysis","🤖 Predictive Maintenance", "⚠️Early Warning System"])
        with t1: render_eda_tab(df_2026,"2026 YTD")
        with t2: render_trend_tab(df_2026,"2026 YTD")
        with t3: render_nlp_tab(df_2026,"2026 YTD")
        with t4: render_prediction_results_tab()
        with t5: render_early_warning_tab(df_2026, "2026 YTD")
    else:
        render_comparative_page(df_hist, df_2026)
