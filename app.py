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
    /* Tab list: centered + bigger */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b27;
        border-radius: 8px 8px 0 0;
        padding: 14px 36px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff22;
        border-bottom: 3px solid #00d4ff;
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
    /* Caption / insight text */
    .stCaption { color: #667788; }
    .insight-box {
        background-color: #161b27;
        border-left: 3px solid #00d4ff;
        border-radius: 0 6px 6px 0;
        padding: 10px 16px;
        margin-bottom: 12px;
        color: #99aabb;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    hr { border-color: #2a3045; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def insight(text):
    """Render a styled insight / explanation box."""
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def fmt_currency(val):
    if val >= 1e9: return f"${val/1e9:.1f}B"
    if val >= 1e6: return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"

def chart_defaults(h=360):
    return dict(template=TEMPLATE, height=h)

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
SEG_ORDER   = ["Low Income", "Mid Income", "High Income"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning data…")
def load_data():
    loan     = pd.read_csv(BASE + "loan.csv")
    customer = pd.read_csv(BASE + "customer.csv")
    lr       = pd.read_csv(BASE + "loan_with_region.csv")
    sr       = pd.read_csv(BASE + "state_region.csv")
    ly       = pd.read_csv(BASE + "loan_count_by_year.csv")

    # ── Loan cleaning ────────────────────────────────────────────────────────
    loan["issue_year"]   = loan["issue_year"].astype(int)
    loan["term"]         = loan["term"].str.strip()
    loan["type"]         = (
        loan["type"].str.strip()
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

    # Simplified loan status (group minor statuses into "Other")
    def status_simplified(s):
        if s in ["Current", "Fully Paid", "Charged Off"]:
            return s
        return "Other"
    loan["loan_status_grouped"] = loan["loan_status"].apply(status_simplified)

    # ── Customer cleaning ─────────────────────────────────────────────────────
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

    centroids = scaler.inverse_transform(km.cluster_centers_).flatten()
    rank      = np.argsort(centroids)
    lbl_map   = {int(rank[0]): "Low Income",
                 int(rank[1]): "Mid Income",
                 int(rank[2]): "High Income"}
    customer["income_segment"] = customer["income_cluster"].map(lbl_map)

    seg_ranges = (
        customer.groupby("income_segment")["annual_inc"]
        .agg(["min", "max"])
        .round(0)
        .astype(int)
    )

    # ── Master merge ──────────────────────────────────────────────────────────
    df = loan.merge(customer, on="customer_id", how="left")
    df = df.merge(lr[["loan_id", "region"]], on="loan_id", how="left")

    ly["issue_year"] = ly["issue_year"].astype(int)

    return df, customer, sr, ly, seg_ranges


df, customer, state_region, loan_year, seg_ranges = load_data()

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
# ║  TAB 1 · OVERVIEW                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[0]:
    st.markdown("### Key Performance Indicators")
    insight(
        "Top-level summary of the entire loan portfolio. "
        "These KPIs provide a quick pulse check — total scale, average cost of credit, "
        "and what share of loans are showing stress signals."
    )

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
        insight(
            "Where do loans stand today? The three main outcomes are "
            "<b>Current</b> (on-time payments), <b>Fully Paid</b> (closed successfully), "
            "and <b>Charged Off</b> (written off as a loss). "
            "<b>Other</b> groups smaller categories: In Grace Period, Late (16–30 days), "
            "Late (31–120 days), and Default.",
        )
        status_df = (
            df["loan_status_grouped"]
            .value_counts()
            .reset_index()
        )
        status_df.columns = ["status", "count"]
        STATUS_COLORS = {
            "Current":     "#2ecc71",
            "Fully Paid":  "#3498db",
            "Charged Off": "#e74c3c",
            "Other":       "#95a5a6",
        }
        fig = px.pie(
            status_df, values="count", names="status",
            hole=0.55,
            color="status", color_discrete_map=STATUS_COLORS,
            **chart_defaults(),
        )
        fig.update_layout(legend=dict(orientation="v"), margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Risk Category Distribution")
        insight(
            "Loans are bucketed into three risk tiers based on repayment behavior. "
            "<b>Not Risky</b>: current, paid, or in grace period. "
            "<b>Risky</b>: 16–30 days late. "
            "<b>Very Risky</b>: 31+ days late, charged off, or defaulted. "
            "The vast majority of loans are on track — the small red bar is where credit losses occur."
        )
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
    insight(
        "Loan grades (A–G) are assigned at issuance based on the borrower's creditworthiness "
        "(credit score, debt-to-income ratio, employment history). Grade A = most creditworthy, "
        "Grade G = highest credit risk. This chart shows whether the assigned grade actually "
        "predicted real-world repayment behavior — it validates the grading system. "
        "A well-calibrated model should show Grade A with near-zero Very Risky, rising steadily toward Grade G."
    )
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
        labels={"pct": "% of Loans in Grade", "grade": "Loan Grade", "risk_category": "Risk Tier"},
        category_orders={"grade": GRADE_ORDER},
        **chart_defaults(),
    )
    fig.update_layout(margin=dict(t=10), legend_title="Risk Tier")
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 · CUSTOMER SEGMENTATION                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[1]:
    st.markdown("### Customer Segmentation by Income")
    insight(
        "Customers are divided into three income groups using K-Means clustering (k=3) "
        "applied to annual income after removing outliers via IQR. "
        "Clusters are ranked by centroid → Low / Mid / High Income. "
        "This segmentation reveals how a borrower's earning power shapes their loan decisions and repayment risk."
    )

    # ── Year selector ─────────────────────────────────────────────────────
    all_years = sorted(df["issue_year"].unique().tolist())
    year_options = ["All Years"] + [str(y) for y in all_years]
    sel_year = st.selectbox(
        "📅 Filter loan behaviour by year:",
        year_options,
        index=0,
        help="Filters all loan behaviour charts below. Income segment definition cards are always based on the full population."
    )

    if sel_year == "All Years":
        seg_df = df.dropna(subset=["income_segment"])
        year_label = "All Years (2012–2019)"
    else:
        seg_df = df[df["issue_year"] == int(sel_year)].dropna(subset=["income_segment"])
        year_label = str(sel_year)

    # ── Segment summary cards ─────────────────────────────────────────────
    st.markdown("#### Income Segment Definitions")
    insight(
        "These ranges are fixed — they reflect the income distribution of the full customer base. "
        "The year filter below applies only to loan behaviour charts, not segment membership."
    )
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
        insight(
            "Box plots show the spread of annual income within each cluster. "
            "The line inside the box is the median; the box covers the middle 50% of incomes. "
            "Wider boxes indicate more income diversity within that segment."
        )
        cust_clean = customer.dropna(subset=["income_segment"])
        fig = px.box(
            cust_clean,
            x="income_segment", y="annual_inc",
            color="income_segment", color_discrete_map=SEG_COLORS,
            category_orders={"income_segment": SEG_ORDER},
            labels={"annual_inc": "Annual Income ($)", "income_segment": ""},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Segment Composition")
        insight(
            "How is the customer base distributed across segments? "
            "A large Low Income share means the platform primarily serves borrowers "
            "with limited income — relevant for understanding default risk exposure."
        )
        seg_cnt = customer["income_segment"].value_counts().reset_index()
        seg_cnt.columns = ["segment", "count"]
        fig = px.pie(
            seg_cnt, values="count", names="segment",
            color="segment", color_discrete_map=SEG_COLORS,
            hole=0.52, **chart_defaults(),
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Loan behaviour section ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### Loan Behaviour by Income Segment — *{year_label}*")
    insight(
        "These charts show how borrowing patterns and credit risk differ across income groups. "
        "Use the year filter above to see how each segment's behaviour evolved over time."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Loan Amount**")
        st.caption("Do higher-income borrowers take larger loans?")
        sa = (
            seg_df.groupby("income_segment")["loan_amount"]
            .mean()
            .reset_index()
            .rename(columns={"loan_amount": "avg_loan"})
        )
        fig = px.bar(
            sa, x="income_segment", y="avg_loan", color="income_segment",
            color_discrete_map=SEG_COLORS,
            category_orders={"income_segment": SEG_ORDER},
            labels={"avg_loan": "Avg Loan ($)", "income_segment": ""},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Very Risky Rate**")
        st.caption("Which income group defaults the most?")
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
            category_orders={"income_segment": SEG_ORDER},
            labels={"very_risky_pct": "Very Risky Rate (%)", "income_segment": ""},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Interest Rate**")
        st.caption("Lower-income borrowers typically carry higher rates due to lower credit grades.")
        si = (
            seg_df.groupby("income_segment")["int_rate_pct"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            si, x="income_segment", y="int_rate_pct", color="income_segment",
            color_discrete_map=SEG_COLORS,
            category_orders={"income_segment": SEG_ORDER},
            labels={"int_rate_pct": "Avg Interest Rate (%)", "income_segment": ""},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 5 Loan Purposes per Segment**")
        st.caption("What are borrowers in each segment actually using loans for?")
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
    insight(
        "Home ownership is a strong proxy for financial stability. "
        "High-income borrowers are more likely to own or have a mortgage, "
        "while lower-income borrowers tend to rent — which can correlate with higher loan risk."
    )
    ho = (
        seg_df.groupby(["income_segment", "home_ownership"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        ho, x="income_segment", y="count", color="home_ownership",
        barmode="group",
        category_orders={"income_segment": SEG_ORDER},
        labels={"count": "Count", "income_segment": "", "home_ownership": "Ownership"},
        **chart_defaults(),
    )
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 · LOAN ANALYSIS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[2]:
    st.markdown("### Loan Analysis")
    insight(
        "Deep dive into the structure of the loan book — how much is being lent, "
        "at what rates, for what purpose, and which categories carry the most credit risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loan Amount Distribution")
        insight(
            "Most loans cluster between $5K–$20K. The sharp peaks at round numbers "
            "($10K, $15K, $20K) reflect borrowers requesting standard amounts. "
            "The right tail shows a minority of high-value loans up to $40K."
        )
        fig = px.histogram(
            df, x="loan_amount", nbins=60,
            color_discrete_sequence=["#00d4ff"],
            labels={"loan_amount": "Loan Amount ($)", "count": "Loans"},
            **chart_defaults(),
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Interest Rate Distribution by Income Segment")
        insight(
            "Interest rates reflect the lender's assessment of credit risk. "
            "Low-income borrowers typically receive higher grades (D–G) due to lower creditworthiness, "
            "resulting in higher rates. This chart shows whether income is a good proxy for rate levels."
        )
        seg_loan = df.dropna(subset=["income_segment"])
        fig = px.box(
            seg_loan, x="income_segment", y="int_rate_pct",
            color="income_segment", color_discrete_map=SEG_COLORS,
            category_orders={"income_segment": SEG_ORDER},
            labels={"int_rate_pct": "Interest Rate (%)", "income_segment": "Income Segment"},
            **chart_defaults(),
        )
        fig.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loan Purpose Breakdown")
        insight(
            "Debt consolidation dominates — borrowers are primarily using personal loans "
            "to refinance existing high-interest debt. Credit card payoff is the #2 reason. "
            "This concentration in a single purpose can be a risk if consumer debt levels rise."
        )
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
        insight(
            "36-month loans are more common — borrowers prefer shorter commitments with "
            "higher monthly payments. 60-month loans carry more interest rate risk for lenders "
            "and typically go to borrowers who need lower monthly payments."
        )
        tc = df["term"].value_counts().reset_index()
        tc.columns = ["term", "count"]
        fig = px.pie(
            tc, values="count", names="term",
            hole=0.52, color_discrete_sequence=["#00d4ff", "#9b59b6"],
            **chart_defaults(),
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Risk Profile by Loan Purpose — Risky & Very Risky Only (Top 10)")
    insight(
        "Showing only the two stress tiers (Risky and Very Risky) removes noise from the "
        "large 'Not Risky' base and makes it easier to compare which loan purposes generate "
        "the most credit problems. Taller bars = higher proportion of troubled loans for that purpose."
    )
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
    # Only show Risky and Very Risky
    pr_stress = pr[pr["risk_category"].isin(["Risky", "Very Risky"])]
    fig = px.bar(
        pr_stress, x="purpose", y="pct", color="risk_category",
        color_discrete_map=RISK_COLORS, barmode="group",
        labels={"pct": "% of Loans in Purpose", "purpose": "", "risk_category": "Risk Tier"},
        height=420, template=TEMPLATE,
    )
    fig.update_layout(
        margin=dict(t=10), xaxis_tickangle=-25,
        legend_title="Risk Tier",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Loan Amount by Income Segment & Purpose (Top 6 Purposes)")
    insight(
        "Breaks down the average loan amount by both who is borrowing (income segment) "
        "and why (purpose). This reveals whether high-income borrowers consistently request "
        "larger loans across all purposes, or if certain purposes attract disproportionately "
        "large requests regardless of income."
    )
    top6_p = df["purpose"].value_counts().head(6).index
    ip = (
        df[df["purpose"].isin(top6_p)]
        .dropna(subset=["income_segment"])
        .groupby(["income_segment", "purpose"])["loan_amount"]
        .mean()
        .reset_index()
    )
    fig = px.bar(
        ip, x="income_segment", y="loan_amount", color="purpose",
        barmode="group",
        category_orders={"income_segment": SEG_ORDER},
        labels={"loan_amount": "Avg Loan Amount ($)", "income_segment": "Income Segment"},
        height=420, template=TEMPLATE,
    )
    fig.update_layout(margin=dict(t=10), legend_title="Loan Purpose")
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 · REGIONAL ANALYSIS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[3]:
    st.markdown("### Regional Analysis")
    insight(
        "Geography matters in credit — state-level economic conditions, unemployment rates, "
        "and local housing markets all influence loan volumes and default rates. "
        "Use the map to spot concentration risk and regional stress patterns."
    )

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

    st.markdown("#### State Map")
    insight(
        "Toggle between <b>Total Loan Volume</b> (where lending activity is concentrated) "
        "and <b>Very Risky Rate</b> (where credit stress is highest). "
        "A state with high volume AND high risk is a double concern for portfolio managers."
    )
    map_metric = st.radio(
        "Select map metric:",
        ["Total Loan Volume ($)", "Very Risky Rate (%)"],
        horizontal=True,
    )

    if map_metric == "Total Loan Volume ($)":
        color_col, color_label = "total_volume", "Loan Volume ($)"
        color_scale = "Blues"
        hover_extra = {"avg_loan": ":,.0f", "total_loans": ":,"}
        title_txt   = "Total Loan Volume by State"
    else:
        color_col, color_label = "very_risky_rate", "Very Risky Rate (%)"
        color_scale = "Reds"
        hover_extra = {"total_loans": ":,", "total_volume": ":,.0f"}
        title_txt   = "Very Risky Loan Rate by State"

    fig = px.choropleth(
        state_agg,
        locations="state", locationmode="USA-states",
        color=color_col, color_continuous_scale=color_scale,
        scope="usa", template=TEMPLATE,
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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 10 States — Loan Volume")
        insight("States with the most lending activity. High volume states like CA, TX, NY drive the portfolio.")
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
        insight("States where the highest proportion of loans are in distress. Smaller states can appear here due to less volume.")
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

    st.markdown("#### Loan Volume vs Very Risky Rate by Region")
    insight(
        "Dual-axis chart comparing lending volume (bars) against credit stress (line) at the regional level. "
        "Regions where the red line peaks despite moderate bar height signal efficiency concerns — "
        "high risk relative to portfolio size."
    )
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
# ║  TAB 5 · TEMPORAL TRENDS                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[4]:
    st.markdown("### Temporal Trends (2012 – 2019)")
    insight(
        "How did the lending platform grow over 8 years? "
        "These charts track volume, loan size, interest rates, and risk mix over time — "
        "revealing market cycles, strategic pivots, and early warning signs of credit deterioration."
    )

    st.markdown("#### Loan Issuance Over Time")
    insight(
        "The platform grew rapidly from 2012 to 2015, then continued scaling. "
        "Year-over-year growth rate is a key indicator of market expansion. "
        "Any sudden dip may signal tightening credit standards or market conditions."
    )
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
        insight(
            "Rising average loan amounts over time can indicate borrowers taking on more debt, "
            "inflation effects, or the platform attracting higher-value customers."
        )
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
        insight(
            "Interest rates are sensitive to macroeconomic conditions (Fed rate changes) "
            "and the platform's credit mix. A rising rate trend alongside rising defaults "
            "is a classic late-cycle credit warning."
        )
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
    insight(
        "How has the risk composition of the portfolio shifted each year? "
        "A growing Very Risky band is an early warning of deteriorating credit quality. "
        "Watch for any year where the red area expands disproportionately."
    )
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
        labels={"pct": "% of Loans", "issue_year": "Year", "risk_category": "Risk Tier"},
        height=380, template=TEMPLATE,
    )
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Loan Grade Mix Over Time")
    insight(
        "Shows whether the platform's credit standards shifted over time. "
        "A growing share of lower grades (E–G) in later years may indicate "
        "loosening underwriting or deliberate expansion into riskier borrower segments."
    )
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
    "Credit Loan Intelligence Dashboard · Built with Streamlit & Plotly · "
    "270K loans · 2012–2019 · United States"
    "</div>",
    unsafe_allow_html=True,
)
