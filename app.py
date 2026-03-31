import warnings
warnings.filterwarnings("ignore")
import os

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Loan Intelligence",
    page_icon="💳",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b27;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff22;
        border-bottom: 2px solid #00d4ff;
    }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #161b27;
        border: 1px solid #2a3045;
        border-radius: 10px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #8899aa; font-size: 0.8rem; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #00d4ff; font-size: 1.6rem; font-weight: 700;
    }
    div[data-testid="metric-container"] div[data-testid="metric-delta"] { color: #aab; }
    /* Section headers */
    h3 { color: #00d4ff; border-bottom: 1px solid #2a3045; padding-bottom: 6px; }
    h4 { color: #c8d8e8; }
    /* Caption text */
    .stCaption { color: #667788; }
    /* Divider */
    hr { border-color: #2a3045; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE = "plotly_dark"
BASE = os.path.join(os.path.dirname(__file__), "data set") + "/"

RISK_COLORS = {
    "Not Risky":  "#2ecc71",
    "Risky":      "#f39c12",
    "Very Risky": "#e74c3c",
}
SEG_COLORS = {
    "Low Income":  "#3498db",
    "Mid Income":  "#9b59b6",
    "High Income": "#1abc9c",
}
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning data…")
def load_data():
    # --- Load ---
    loan     = pd.read_csv(BASE + "loan.csv")
    customer = pd.read_csv(BASE + "customer.csv")
    lr       = pd.read_csv(BASE + "loan_with_region.csv")
    sr       = pd.read_csv(BASE + "state_region.csv")
    ly       = pd.read_csv(BASE + "loan_count_by_year.csv")

    # ── Loan cleaning ────────────────────────────────────────────────────────
    loan["issue_year"] = loan["issue_year"].astype(int)
    loan["term"]       = loan["term"].str.strip()
    loan["type"]       = (
        loan["type"]
        .str.strip()
        .replace({"Individual": "INDIVIDUAL", "Joint App": "JOINT"})
    )
    loan["int_rate_pct"] = loan["int_rate"] * 100

    # Risk category (3 tiers)
    def risk_cat(s):
        if s in ["Current", "Fully Paid", "In Grace Period"]:
            return "Not Risky"
        if s in ["Late (16-30 days)"]:
            return "Risky"
        return "Very Risky"   # Late (31-120), Charged Off, Default

    loan["risk_category"] = loan["loan_status"].apply(risk_cat)

    # ── Customer cleaning ─────────────────────────────────────────────────────
    # IQR outlier removal on annual income
    q1, q3 = customer["annual_inc"].quantile([0.25, 0.75])
    iqr     = q3 - q1
    customer = customer[
        customer["annual_inc"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    ].copy()

    # K-Means income segmentation (k=3)
    inc_df  = customer[["annual_inc"]].dropna()
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(inc_df)
    km      = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(scaled)
    customer.loc[inc_df.index, "income_cluster"] = km.labels_

    # Map cluster IDs → Low / Mid / High by centroid value
    centroids = scaler.inverse_transform(km.cluster_centers_).flatten()
    rank      = np.argsort(centroids)           # ascending order
    lbl_map   = {int(rank[0]): "Low Income",
                 int(rank[1]): "Mid Income",
                 int(rank[2]): "High Income"}
    customer["income_segment"] = customer["income_cluster"].map(lbl_map)

    # Segment dollar ranges (for display)
    seg_ranges = (
        customer.groupby("income_segment")["annual_inc"]
        .agg(["min", "max"])
        .round(0)
        .astype(int)
    )

    # ── Merge all into one master df ──────────────────────────────────────────
    df = loan.merge(customer, on="customer_id", how="left")
    df = df.merge(lr[["loan_id", "region"]], on="loan_id", how="left")

    # Year table
    ly["issue_year"] = ly["issue_year"].astype(int)

    return df, customer, sr, ly, seg_ranges


df, customer, state_region, loan_year, seg_ranges = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_currency(val):
    if val >= 1e9: return f"${val/1e9:.1f}B"
    if val >= 1e6: return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"

def chart_defaults():
    return dict(template=TEMPLATE, height=360)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:28px 0 12px 0;'>
    <span style='font-size:2.6rem; font-weight:800; color:#00d4ff;'>💳 Credit Loan Intelligence</span><br>
    <span style='color:#6688aa; font-size:1rem; letter-spacing:1px;'>
        270 000 loans &nbsp;·&nbsp; 2012 – 2019 &nbsp;·&nbsp; United States
    </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "👥 Customer Segmentation",
    "📊 Loan Analysis",
    "🗺️ Regional Analysis",
    "📈 Temporal Trends",
])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 · OVERVIEW                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[0]:
    st.markdown("### Key Performance Indicators")

    total_loans    = len(df)
    total_vol      = df["loan_amount"].sum()
    avg_int        = df["int_rate_pct"].mean()
    very_risky_pct = (df["risk_category"] == "Very Risky").mean() * 100
    risky_pct      = (df["risk_category"] == "Risky").mean() * 100
    avg_loan       = df["loan_amount"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Loans",       f"{total_loans:,}")
    c2.metric("Total Volume",      fmt_currency(total_vol))
    c3.metric("Avg Loan Amount",   fmt_currency(avg_loan))
    c4.metric("Avg Interest Rate", f"{avg_int:.1f}%")
    c5.metric("Very Risky Rate",   f"{very_risky_pct:.1f}%")
    c6.metric("Risky Rate",        f"{risky_pct:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loan Status Breakdown")
        status_df = df["loan_status"].value_counts().reset_index()
        status_df.columns = ["status", "count"]
        fig = px.pie(
            status_df, values="count", names="status",
            hole=0.55, **chart_defaults(),
        )
        fig.update_layout(legend=dict(orientation="v"), margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Risk Category Distribution")
        risk_df = df["risk_category"].value_counts().reset_index()
        risk_df.columns = ["risk", "count"]
        risk_df["pct"] = (risk_df["count"] / len(df) * 100).round(1)
        fig = px.bar(
            risk_df, x="risk", y="count", color="risk",
            color_discrete_map=RISK_COLORS,
            text=risk_df["pct"].apply(lambda x: f"{x}%"),
            **chart_defaults(),
        )
        fig.update_layout(
            showlegend=False, xaxis_title="", yaxis_title="Loan Count",
            margin=dict(t=20),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Risk Profile by Loan Grade")
    st.caption("Grade A = lowest risk/interest · Grade G = highest risk/interest")
    grade_risk = (
        df.groupby(["grade", "risk_category"])
        .size()
        .reset_index(name="count")
    )
    grade_total = df.groupby("grade").size().reset_index(name="total")
    grade_risk  = grade_risk.merge(grade_total, on="grade")
    grade_risk["pct"] = grade_risk["count"] / grade_risk["total"] * 100
    fig = px.bar(
        grade_risk, x="grade", y="pct", color="risk_category",
        color_discrete_map=RISK_COLORS, barmode="stack",
        labels={"pct": "%", "grade": "Grade", "risk_category": "Risk"},
        category_orders={"grade": GRADE_ORDER},
        **chart_defaults(),
    )
    fig.update_layout(margin=dict(t=10), legend_title="Risk Category")
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 · CUSTOMER SEGMENTATION                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[1]:
    st.markdown("### Customer Segmentation by Income")
    st.caption(
        "K-Means (k=3) applied to annual income after IQR outlier removal. "
        "Clusters ranked by centroid value → Low / Mid / High Income."
    )

    # ── Segment summary cards ─────────────────────────────────────────────
    st.markdown("#### Income Segments")
    sc1, sc2, sc3 = st.columns(3)
    for col, seg in zip([sc1, sc2, sc3], ["Low Income", "Mid Income", "High Income"]):
        if seg in seg_ranges.index:
            lo    = seg_ranges.loc[seg, "min"]
            hi    = seg_ranges.loc[seg, "max"]
            cnt   = (customer["income_segment"] == seg).sum()
            share = cnt / len(customer) * 100
            col.metric(
                seg,
                f"{cnt:,} customers ({share:.0f}%)",
                f"${lo:,} – ${hi:,} / year",
            )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Income Distribution per Segment")
        cust_clean = customer.dropna(subset=["income_segment"])
        fig = px.box(
            cust_clean,
            x="income_segment", y="annual_inc",
            color="income_segment", color_discrete_map=SEG_COLORS,
            category_orders={"income_segment": ["Low Income", "Mid Income", "High Income"]},
            labels={"annual_inc": "Annual Income ($)", "income_segment": ""},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Segment Composition")
        seg_cnt = customer["income_segment"].value_counts().reset_index()
        seg_cnt.columns = ["segment", "count"]
        fig = px.pie(
            seg_cnt, values="count", names="segment",
            color="segment", color_discrete_map=SEG_COLORS,
            hole=0.52, **chart_defaults(),
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Loan behaviour by segment ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Loan Behaviour by Income Segment")

    seg_df = df.dropna(subset=["income_segment"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Loan Amount**")
        sa = (
            seg_df.groupby("income_segment")["loan_amount"]
            .mean()
            .reset_index()
            .rename(columns={"loan_amount": "avg_loan"})
        )
        fig = px.bar(
            sa, x="income_segment", y="avg_loan", color="income_segment",
            color_discrete_map=SEG_COLORS,
            category_orders={"income_segment": ["Low Income", "Mid Income", "High Income"]},
            labels={"avg_loan": "Avg Loan ($)", "income_segment": ""},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Very Risky Rate**")
        sr2 = (
            seg_df.assign(vr=seg_df["risk_category"].eq("Very Risky"))
            .groupby("income_segment")["vr"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"vr": "very_risky_pct"})
        )
        fig = px.bar(
            sr2, x="income_segment", y="very_risky_pct", color="income_segment",
            color_discrete_map=SEG_COLORS,
            category_orders={"income_segment": ["Low Income", "Mid Income", "High Income"]},
            labels={"very_risky_pct": "Very Risky Rate (%)", "income_segment": ""},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Interest Rate**")
        si = (
            seg_df.groupby("income_segment")["int_rate_pct"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            si, x="income_segment", y="int_rate_pct", color="income_segment",
            color_discrete_map=SEG_COLORS,
            category_orders={"income_segment": ["Low Income", "Mid Income", "High Income"]},
            labels={"int_rate_pct": "Avg Interest Rate (%)", "income_segment": ""},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 5 Loan Purposes per Segment**")
        tp = (
            seg_df.groupby(["income_segment", "purpose"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .groupby("income_segment")
            .head(5)
        )
        fig = px.bar(
            tp, x="count", y="purpose", color="income_segment",
            color_discrete_map=SEG_COLORS, orientation="h",
            facet_col="income_segment",
            labels={"count": "Loans", "purpose": ""},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=40))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Home Ownership Mix by Segment")
    ho = (
        seg_df.groupby(["income_segment", "home_ownership"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        ho, x="income_segment", y="count", color="home_ownership",
        barmode="group",
        category_orders={"income_segment": ["Low Income", "Mid Income", "High Income"]},
        labels={"count": "Count", "income_segment": "", "home_ownership": "Ownership"},
        **chart_defaults(),
    )
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 · LOAN ANALYSIS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[2]:
    st.markdown("### Loan Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loan Amount Distribution")
        fig = px.histogram(
            df, x="loan_amount", nbins=60,
            color_discrete_sequence=["#00d4ff"],
            labels={"loan_amount": "Loan Amount ($)", "count": "Loans"},
            **chart_defaults(),
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Interest Rate Distribution by Grade")
        fig = px.box(
            df, x="grade", y="int_rate_pct", color="grade",
            category_orders={"grade": GRADE_ORDER},
            labels={"int_rate_pct": "Interest Rate (%)", "grade": "Grade"},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loan Purpose Breakdown")
        pc = df["purpose"].value_counts().reset_index()
        pc.columns = ["purpose", "count"]
        fig = px.treemap(
            pc, path=["purpose"], values="count",
            color="count", color_continuous_scale="Blues",
            **chart_defaults(),
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Loan Term Split")
        tc = df["term"].value_counts().reset_index()
        tc.columns = ["term", "count"]
        fig = px.pie(
            tc, values="count", names="term",
            hole=0.52, color_discrete_sequence=["#00d4ff", "#9b59b6"],
            **chart_defaults(),
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Risk Profile by Loan Purpose (Top 10)")
    st.caption("Shows % split of risk tiers for the 10 most common loan purposes")
    top10_p = df["purpose"].value_counts().head(10).index
    pr = (
        df[df["purpose"].isin(top10_p)]
        .groupby(["purpose", "risk_category"])
        .size()
        .reset_index(name="count")
    )
    pt = df[df["purpose"].isin(top10_p)].groupby("purpose").size().reset_index(name="total")
    pr = pr.merge(pt, on="purpose")
    pr["pct"] = pr["count"] / pr["total"] * 100
    fig = px.bar(
        pr, x="purpose", y="pct", color="risk_category",
        color_discrete_map=RISK_COLORS, barmode="stack",
        labels={"pct": "% of Loans", "purpose": "", "risk_category": "Risk"},
        height=400, template=TEMPLATE,
    )
    fig.update_layout(margin=dict(t=10), xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Loan Amount by Grade & Purpose (Top 6 Purposes)")
    top6_p = df["purpose"].value_counts().head(6).index
    gp = (
        df[df["purpose"].isin(top6_p)]
        .groupby(["grade", "purpose"])["loan_amount"]
        .mean()
        .reset_index()
    )
    fig = px.line(
        gp, x="grade", y="loan_amount", color="purpose",
        markers=True,
        category_orders={"grade": GRADE_ORDER},
        labels={"loan_amount": "Avg Loan Amount ($)", "grade": "Grade"},
        height=400, template=TEMPLATE,
    )
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 · REGIONAL ANALYSIS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[3]:
    st.markdown("### Regional Analysis")

    # ── State-level aggregation ───────────────────────────────────────────
    state_agg = (
        df.groupby("state")
        .agg(
            total_volume=("loan_amount", "sum"),
            total_loans=("loan_id", "count"),
            very_risky=("risk_category", lambda x: (x == "Very Risky").sum()),
        )
        .reset_index()
    )
    state_agg["very_risky_rate"] = (
        state_agg["very_risky"] / state_agg["total_loans"] * 100
    )
    state_agg["avg_loan"] = state_agg["total_volume"] / state_agg["total_loans"]

    # ── Map selector ──────────────────────────────────────────────────────
    st.markdown("#### State Map")
    map_metric = st.radio(
        "Select map metric:",
        ["Total Loan Volume ($)", "Very Risky Rate (%)"],
        horizontal=True,
    )

    if map_metric == "Total Loan Volume ($)":
        color_col   = "total_volume"
        color_label = "Loan Volume ($)"
        color_scale = "Blues"
        hover_extra = {"avg_loan": ":,.0f", "total_loans": ":,"}
        title_txt   = "Total Loan Volume by State"
    else:
        color_col   = "very_risky_rate"
        color_label = "Very Risky Rate (%)"
        color_scale = "Reds"
        hover_extra = {"total_loans": ":,", "total_volume": ":,.0f"}
        title_txt   = "Very Risky Loan Rate by State"

    fig = px.choropleth(
        state_agg,
        locations="state",
        locationmode="USA-states",
        color=color_col,
        color_continuous_scale=color_scale,
        scope="usa",
        template=TEMPLATE,
        labels={color_col: color_label},
        hover_data=hover_extra,
        title=title_txt,
    )
    fig.update_layout(
        geo=dict(bgcolor="#0e1117", lakecolor="#0e1117", landcolor="#1a2035"),
        margin=dict(t=50, b=0),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Top / Bottom tables side by side ──────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 10 States — Loan Volume")
        top_vol = (
            state_agg.nlargest(10, "total_volume")
            [["state", "total_volume", "total_loans", "very_risky_rate"]]
            .reset_index(drop=True)
        )
        fig = px.bar(
            top_vol, x="total_volume", y="state", orientation="h",
            color="total_volume", color_continuous_scale="Blues",
            labels={"total_volume": "Loan Volume ($)", "state": ""},
            height=380, template=TEMPLATE,
        )
        fig.update_layout(
            coloraxis_showscale=False, showlegend=False,
            margin=dict(t=10), yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Top 10 States — Very Risky Rate")
        top_risk = (
            state_agg.nlargest(10, "very_risky_rate")
            [["state", "very_risky_rate", "total_loans"]]
            .reset_index(drop=True)
        )
        fig = px.bar(
            top_risk, x="very_risky_rate", y="state", orientation="h",
            color="very_risky_rate", color_continuous_scale="Reds",
            labels={"very_risky_rate": "Very Risky Rate (%)", "state": ""},
            height=380, template=TEMPLATE,
        )
        fig.update_layout(
            coloraxis_showscale=False, showlegend=False,
            margin=dict(t=10), yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Region-level dual axis ────────────────────────────────────────────
    st.markdown("#### Loan Volume vs Very Risky Rate by Region")
    region_agg = (
        df.groupby("region")
        .agg(
            total_volume=("loan_amount", "sum"),
            total_loans=("loan_id", "count"),
            very_risky=("risk_category", lambda x: (x == "Very Risky").sum()),
        )
        .reset_index()
    )
    region_agg["very_risky_rate"] = (
        region_agg["very_risky"] / region_agg["total_loans"] * 100
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=region_agg["region"], y=region_agg["total_volume"],
            name="Loan Volume", marker_color="#00d4ff", opacity=0.85,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=region_agg["region"], y=region_agg["very_risky_rate"],
            name="Very Risky Rate %", mode="lines+markers",
            marker_color="#e74c3c", line_width=3, marker_size=10,
        ),
        secondary_y=True,
    )
    fig.update_layout(template=TEMPLATE, height=380, margin=dict(t=10))
    fig.update_yaxes(title_text="Loan Volume ($)", secondary_y=False)
    fig.update_yaxes(title_text="Very Risky Rate (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 5 · TEMPORAL TRENDS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[4]:
    st.markdown("### Temporal Trends (2012 – 2019)")

    # Loan count from dedicated table
    st.markdown("#### Loan Issuance Over Time")
    fig = px.area(
        loan_year, x="issue_year", y="loan_count",
        color_discrete_sequence=["#00d4ff"],
        markers=True,
        labels={"loan_count": "Number of Loans", "issue_year": "Year"},
        height=350, template=TEMPLATE,
    )
    fig.update_traces(fill="tozeroy", fillcolor="rgba(0,212,255,0.15)")
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average Loan Amount by Year")
        ya = df.groupby("issue_year")["loan_amount"].mean().reset_index()
        fig = px.line(
            ya, x="issue_year", y="loan_amount",
            markers=True, color_discrete_sequence=["#1abc9c"],
            labels={"loan_amount": "Avg Loan Amount ($)", "issue_year": "Year"},
            height=340, template=TEMPLATE,
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Average Interest Rate by Year")
        yi = df.groupby("issue_year")["int_rate_pct"].mean().reset_index()
        fig = px.line(
            yi, x="issue_year", y="int_rate_pct",
            markers=True, color_discrete_sequence=["#f39c12"],
            labels={"int_rate_pct": "Avg Interest Rate (%)", "issue_year": "Year"},
            height=340, template=TEMPLATE,
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Risk Category Mix Over Time")
    st.caption("Stacked area shows the proportion of each risk tier per year")
    yr = (
        df.groupby(["issue_year", "risk_category"])
        .size()
        .reset_index(name="count")
    )
    yt = df.groupby("issue_year").size().reset_index(name="total")
    yr = yr.merge(yt, on="issue_year")
    yr["pct"] = yr["count"] / yr["total"] * 100
    fig = px.area(
        yr, x="issue_year", y="pct", color="risk_category",
        color_discrete_map=RISK_COLORS,
        labels={"pct": "% of Loans", "issue_year": "Year", "risk_category": "Risk"},
        height=380, template=TEMPLATE,
    )
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Loan Grade Mix Over Time")
    yg = (
        df.groupby(["issue_year", "grade"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        yg, x="issue_year", y="count", color="grade",
        barmode="stack",
        category_orders={"grade": GRADE_ORDER},
        labels={"count": "Loan Count", "issue_year": "Year"},
        height=380, template=TEMPLATE,
    )
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#445566; font-size:0.8rem;'>"
    "Credit Loan Intelligence Dashboard · Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True,
)
