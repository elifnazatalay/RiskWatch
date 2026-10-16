"""RiskWatch fraud risk-monitoring dashboard.

Reads the precomputed files in reports/, so it runs without the raw data or
trained models (that's what lets it deploy to Streamlit Community Cloud).
To regenerate them, run the pipeline in the README, then:
    streamlit run app/dashboard.py
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from riskwatch.config import PROJECT_ROOT
from riskwatch.threshold import threshold_table

REPORTS = PROJECT_ROOT / "reports"
DATA = REPORTS / "dashboard"
MODEL_LABELS = {"random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}

# Categorical slots 1-2 (validated for color-vision deficiency, light and dark).
SERIES_COLORS = {
    "light": {"random_forest": "#2a78d6", "logistic_regression": "#eb6834"},
    "dark": {"random_forest": "#3987e5", "logistic_regression": "#d95926"},
}

st.set_page_config(page_title="RiskWatch", page_icon="🛡️", layout="wide")


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:
        return "light"


def colors() -> dict:
    return SERIES_COLORS[theme()]


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


@st.cache_data
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data
def read_json(path: str) -> dict:
    return json.loads((REPORTS / path).read_text())


def bar(x, y, hover_label: str, y_title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=x, y=y, marker=dict(color=color, cornerradius=4),
        hovertemplate=f"%{{x}}<br>{hover_label}: %{{y:.3f}}%<extra></extra>",
    ))
    fig.update_yaxes(title=y_title, ticksuffix="%")
    return style(fig)


# ---------------------------------------------------------------- header
st.title("🛡️ RiskWatch")
st.caption("Fraud detection and risk monitoring on 284,807 credit-card transactions "
           "(Kaggle / ULB dataset, two days of European cardholder data). "
           "[Source code](https://github.com/elifnazatalay/RiskWatch)")

if not (DATA / "test_scores.csv.gz").exists():
    st.error("Dashboard data is missing. Run `python scripts/compare_models.py`, "
             "`python scripts/benchmark_latency.py` and "
             "`python scripts/export_dashboard_data.py`, then reload.")
    st.stop()

test = read_csv("test_scores.csv.gz")
summary = read_json("dashboard/summary.json")
comparison = read_json("model_comparison.json")
latency = read_json("latency.json")

tab_overview, tab_model, tab_monitor = st.tabs(["Portfolio risk", "Model & threshold", "Monitoring"])

# ---------------------------------------------------------------- tab 1
with tab_overview:
    ov = read_csv("overview.csv").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{int(ov['total_transactions']):,}")
    c2.metric("Fraud transactions", f"{int(ov['fraud_transactions']):,}",
              help="Class = 1 in the dataset.")
    c3.metric("Fraud rate", f"{ov['fraud_rate_pct']:.3f}%")
    c4.metric("Fraud losses", f"${ov['fraud_amount']:,.0f}",
              help=f"{ov['fraud_amount_pct']:.2f}% of total transaction value.")

    left, right = st.columns(2)
    bands = read_csv("amount_bands.csv")
    with left:
        st.subheader("Fraud rate by transaction amount")
        st.plotly_chart(bar(bands["amount_band"].str[3:], bands["fraud_rate_pct"],
                            "Fraud rate", "Fraud rate", colors()["random_forest"]),
                        width="stretch")
        st.caption("$0 transactions are 8.6x riskier than average, consistent with "
                   "card-testing charges.")
    hourly = read_csv("hourly_risk.csv").sort_values("hour")
    with right:
        st.subheader("Fraud rate by hour of data collection")
        st.plotly_chart(bar(hourly["hour"], hourly["fraud_rate_pct"],
                            "Fraud rate", "Fraud rate", colors()["random_forest"]),
                        width="stretch")
        st.caption("Hours count from the first transaction, not clock time. "
                   "Low-volume hours carry the highest fraud rates.")

# ---------------------------------------------------------------- tab 2
with tab_model:
    picked = st.radio("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get,
                      horizontal=True)
    scores = test[f"score_{picked}"]
    chosen = comparison["results"][picked]["threshold"]

    # Logistic Regression's useful thresholds sit above 0.99, so the grid is finer there.
    # The recommended threshold is added unrounded so the slider reproduces the report exactly.
    grid = np.unique(np.concatenate([
        np.round(np.arange(0.01, 0.99, 0.01), 2), np.round(np.arange(0.990, 1.0, 0.001), 3),
        [chosen]]))
    threshold = st.select_slider(
        "Decision threshold: transactions scoring at or above this are flagged",
        options=[float(g) for g in grid], value=float(chosen),
        format_func=lambda v: f"{v:.3f}",
    )
    st.caption(f"Recommended threshold **{chosen:.3f}** keeps validation false alarms at "
               f"or below {comparison['max_validation_fpr']:.1%} of legitimate transactions.")

    row = threshold_table(test["Class"], scores, test["Amount"], [threshold]).iloc[0]
    total_fraud = int(test["Class"].sum())
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Frauds caught", f"{int(row['fraud_caught'])} / {total_fraud}",
              help="Recall on the test period.")
    m2.metric("Recall", f"{row['recall']:.1%}")
    m3.metric("False alarms", f"{int(row['false_alarms']):,}",
              help="Legitimate transactions flagged for review.")
    m4.metric("Precision", f"{row['precision']:.1%}",
              help="Share of alerts that are real fraud.")
    m5.metric("Fraud $ caught", f"{row['fraud_dollars_caught_pct']:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Precision-recall: both models")
        fig = go.Figure()
        for name, label in MODEL_LABELS.items():
            p, r, _ = precision_recall_curve(test["Class"], test[f"score_{name}"])
            pr_auc = comparison["results"][name]["pr_auc"]
            fig.add_trace(go.Scatter(
                x=r, y=p, mode="lines", name=f"{label} (PR-AUC {pr_auc:.3f})",
                line=dict(color=colors()[name], width=2),
                hovertemplate="Recall %{x:.1%}<br>Precision %{y:.1%}<extra>" + label + "</extra>",
            ))
        fig.add_trace(go.Scatter(
            x=[row["recall"]], y=[row["precision"]], mode="markers", name="Current threshold",
            marker=dict(size=11, color=colors()[picked], line=dict(width=2, color="white")),
            hovertemplate="Current threshold<br>Recall %{x:.1%}<br>Precision %{y:.1%}<extra></extra>",
        ))
        fig.update_xaxes(title="Recall (share of fraud caught)", tickformat=".0%", range=[0, 1.02])
        fig.update_yaxes(title="Precision (share of alerts that are fraud)", tickformat=".0%",
                         range=[0, 1.02])
        st.plotly_chart(style(fig, 380), width="stretch")

    with right:
        st.subheader(f"Trade-off: {MODEL_LABELS[picked]}")
        curve = threshold_table(test["Class"], scores, test["Amount"], grid)
        fig = go.Figure(go.Scatter(
            x=curve["false_alarms"], y=curve["fraud_caught"], mode="lines",
            line=dict(color=colors()[picked], width=2), name=MODEL_LABELS[picked],
            customdata=curve["threshold"],
            hovertemplate=("Threshold %{customdata:.3f}<br>%{x:,} false alarms"
                           "<br>%{y} frauds caught<extra></extra>"),
        ))
        fig.add_trace(go.Scatter(
            x=[row["false_alarms"]], y=[row["fraud_caught"]], mode="markers",
            marker=dict(size=11, color=colors()[picked], line=dict(width=2, color="white")),
            name="Current threshold",
            hovertemplate="Current threshold<br>%{x:,} false alarms<br>%{y} caught<extra></extra>",
        ))
        fig.update_xaxes(title="False alarms (log scale)", type="log")
        fig.update_yaxes(title=f"Frauds caught (of {total_fraud})")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 380), width="stretch")
        st.caption("Each step right means more customers reviewed for each extra fraud caught.")

    st.subheader("Model comparison at the recommended thresholds")
    table = pd.DataFrame([
        {
            "Model": MODEL_LABELS[name],
            "PR-AUC": r["pr_auc"],
            "Threshold": r["threshold"],
            "Frauds caught": f"{r['true_positives']} / {r['true_positives'] + r['false_negatives']}",
            "Recall": r["recall"],
            "Precision": r["precision"],
            "False alarms": r["false_positives"],
            "False-positive rate": r["false_positive_rate"],
            "Latency, 1k txns (ms)": latency["results"][name]["median_ms"],
        }
        for name, r in comparison["results"].items()
    ])
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Threshold": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="percent"),
            "Precision": st.column_config.NumberColumn(format="percent"),
            "False-positive rate": st.column_config.NumberColumn(format="%.4f"),
            "Latency, 1k txns (ms)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

# ---------------------------------------------------------------- tab 3
with tab_monitor:
    st.subheader("Data-quality checks (raw CSV)")
    for check in summary["quality_checks"]:
        icon = "✅" if check["passed"] else "❌"
        st.markdown(f"{icon} **{check['name'].replace('_', ' ').capitalize()}** "
                    f"({'pass' if check['passed'] else 'fail'}): {check['detail']}")
    st.caption("Duplicates are reported rather than failed, then removed before modeling.")

    st.subheader("Drift: training period vs. test period")
    st.caption("Population Stability Index (PSI): < 0.10 stable, 0.10-0.25 moderate shift, "
               "> 0.25 significant shift.")
    # Score PSI isn't meaningful here: most transactions score exactly 0, which
    # collapses the quantile bins. The alert rate at the chosen threshold is the
    # number an operations team would actually watch.
    rf_threshold = comparison["results"]["random_forest"]["threshold"]
    val_alert_rate = summary["rf_alert_rate_validation"]
    test_alert_rate = (test["score_random_forest"] >= rf_threshold).mean()
    d1, d2, d3 = st.columns(3)
    d1.metric("Random Forest alert rate, test", f"{test_alert_rate:.3%}",
              delta=f"{(test_alert_rate - val_alert_rate) * 100:+.3f} pts vs validation",
              delta_color="off")
    d2.metric("Fraud rate, train period", f"{summary['fraud_rate_train']:.3%}")
    d3.metric("Fraud rate, test period", f"{summary['fraud_rate_test']:.3%}")

    drift = read_csv("feature_drift.csv")
    top = drift.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["psi"], y=top["feature"], orientation="h",
        marker=dict(color=colors()["random_forest"], cornerradius=4),
        customdata=top["status"],
        hovertemplate="%{y}<br>PSI %{x:.3f} (%{customdata})<extra></extra>",
    ))
    for x, text in [(0.10, "moderate"), (0.25, "significant")]:
        fig.add_vline(x=x, line=dict(dash="dot", width=1, color="gray"),
                      annotation_text=text, annotation_position="top")
    fig.update_xaxes(title="PSI (top 10 features)")
    st.subheader("Feature drift")
    st.plotly_chart(style(fig, 360), width="stretch")
    st.caption("The test period covers the last hours of day 2, while training spans other "
               "hours of the day, so part of this drift may reflect the daily cycle of card "
               "activity rather than new fraud behavior.")
    with st.expander("All features"):
        st.dataframe(drift, hide_index=True, width="stretch",
                     column_config={"psi": st.column_config.NumberColumn(format="%.4f")})

    st.subheader("Inference latency")
    l1, l2 = st.columns(2)
    for col, name in zip([l1, l2], MODEL_LABELS):
        r = latency["results"][name]
        col.metric(f"{MODEL_LABELS[name]}: {r['batch_size']:,} transactions",
                   f"{r['median_ms']:.1f} ms", help=f"median; p95 {r['p95_ms']:.1f} ms")
    st.caption(f"Measured on {latency['machine']} with `scripts/benchmark_latency.py`.")
