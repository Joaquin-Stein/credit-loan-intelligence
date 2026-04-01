import warnings
warnings.filterwarnings("ignore")
import os

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
# ML — LOAN PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
INCOME_OPTIONS = {
    "Under $30,000":       20_000,
    "$30,000 – $50,000":   40_000,
    "$50,000 – $75,000":   62_500,
    "$75,000 – $100,000":  87_500,
    "$100,000 – $150,000": 125_000,
    "Over $150,000":       160_000,
}
INCOME_BINS   = [0, 30_000, 50_000, 75_000, 100_000, 150_000, 1e9]
INCOME_LABELS = list(INCOME_OPTIONS.keys())

EMP_OPTIONS = {
    "Less than 1 year": 0,
    "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8,
    "9 years": 9, "10+ years": 10,
}

PURPOSE_DISPLAY = {
    "debt_consolidation": "Debt Consolidation",
    "credit_card":        "Credit Card Payoff",
    "home_improvement":   "Home Improvement",
    "other":              "Other",
    "major_purchase":     "Major Purchase",
    "medical":            "Medical",
    "small_business":     "Small Business",
    "car":                "Car",
    "vacation":           "Vacation",
    "moving":             "Moving / Relocation",
    "house":              "House",
    "wedding":            "Wedding",
    "renewable_energy":   "Renewable Energy",
    "educational":        "Educational",
}

HOME_OPTIONS = ["MORTGAGE", "RENT", "OWN"]
HOME_DISPLAY = {"MORTGAGE": "Mortgage", "RENT": "Rent", "OWN": "Own"}


@st.cache_resource(show_spinner="Training prediction models…")
def train_models(_df):
    """
    Hybrid approach:
      - Loan Amount  : GradientBoosting regression  (predicted from user inputs)
      - Interest Rate: historical percentile lookup      (p25 / median / p75 for similar borrowers)
    Interest rate is almost entirely determined by credit grade (FICO-based), which is not
    available as a user input — so regression adds no value.  Percentile lookup from 270K
    real loans is far more informative and honest.
    """

    def parse_emp(s):
        if pd.isna(s) or str(s).strip().lower() in ["n/a", "nan", ""]:
            return 3
        s = str(s)
        if "10+" in s:
            return 10
        if "< 1" in s or "<1" in s:
            return 0
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 3

    cols = ["loan_amount", "int_rate_pct", "annual_inc", "emp_length",
            "purpose", "home_ownership", "avg_cur_bal", "Tot_cur_bal"]
    ml = _df[cols].copy()
    ml["emp_length_yrs"] = ml["emp_length"].apply(parse_emp)
    ml = ml.drop(columns=["emp_length"])
    ml = ml.dropna()
    ml = ml[ml["home_ownership"].isin(HOME_OPTIONS)]

    NUMERIC  = ["annual_inc", "emp_length_yrs", "avg_cur_bal", "Tot_cur_bal"]
    CATEG    = ["purpose", "home_ownership"]

    # ── Loan amount model ─────────────────────────────────────────────────
    X = ml[NUMERIC + CATEG]
    y_amt = ml["loan_amount"]
    X_tr, X_te, ya_tr, ya_te = train_test_split(X, y_amt, test_size=0.15, random_state=42)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEG),
    ], remainder="passthrough")
    pipe_amt = Pipeline([
        ("pre", pre),
        ("model", GradientBoostingRegressor(
            n_estimators=120, max_depth=5, learning_rate=0.1,
            min_samples_leaf=30, random_state=42, subsample=0.8
        )),
    ])
    pipe_amt.fit(X_tr, ya_tr)
    r2_amt = r2_score(ya_te, pipe_amt.predict(X_te))

    # ── Rate lookup table: income_bucket × purpose ────────────────────────
    ml["inc_label"] = pd.cut(ml["annual_inc"], bins=INCOME_BINS, labels=INCOME_LABELS)
    rate_lookup = (
        ml.groupby(["inc_label", "purpose"], observed=True)["int_rate_pct"]
        .agg(rate_p25=lambda x: x.quantile(0.25),
             rate_med=lambda x: x.quantile(0.50),
             rate_p75=lambda x: x.quantile(0.75),
             n="count")
        .reset_index()
    )
    # Fallback: income-only lookup (when purpose bucket is thin)
    rate_lookup_inc = (
        ml.groupby("inc_label", observed=True)["int_rate_pct"]
        .agg(rate_p25=lambda x: x.quantile(0.25),
             rate_med=lambda x: x.quantile(0.50),
             rate_p75=lambda x: x.quantile(0.75))
        .reset_index()
    )

    # ── Background features (median balances per income bucket) ───────────
    bg = ml.groupby("inc_label", observed=True)[["avg_cur_bal", "Tot_cur_bal"]].median()

    # ── Feature importance for loan amount model ──────────────────────────
    cat_features = (pipe_amt.named_steps["pre"]
                    .named_transformers_["cat"]
                    .get_feature_names_out(CATEG).tolist())
    feat_names = cat_features + NUMERIC
    importance = pipe_amt.named_steps["model"].feature_importances_

    return pipe_amt, r2_amt, rate_lookup, rate_lookup_inc, bg, feat_names, importance


def get_rate_range(rate_lookup, rate_lookup_inc, inc_label, purpose):
    """Return (p25, median, p75) interest rate for the given profile."""
    row = rate_lookup[
        (rate_lookup["inc_label"] == inc_label) &
        (rate_lookup["purpose"]   == purpose)
    ]
    if len(row) > 0 and float(row["n"].iloc[0]) >= 30:
        r = row.iloc[0]
        return float(r["rate_p25"]), float(r["rate_med"]), float(r["rate_p75"])
    # Fallback: income bucket only
    row2 = rate_lookup_inc[rate_lookup_inc["inc_label"] == inc_label]
    if len(row2) > 0:
        r = row2.iloc[0]
        return float(r["rate_p25"]), float(r["rate_med"]), float(r["rate_p75"])
    return 10.0, 13.5, 18.0   # global fallback


def calc_installment(loan_amt, int_rate_pct, months):
    r = (int_rate_pct / 100) / 12
    if r == 0:
        return loan_amt / months
    return loan_amt * (r * (1 + r) ** months) / ((1 + r) ** months - 1)


def profile_label(int_rate_pct):
    if int_rate_pct < 8:   return "Excellent",      "#2ecc71"
    if int_rate_pct < 12:  return "Good",            "#1abc9c"
    if int_rate_pct < 18:  return "Average",         "#f39c12"
    if int_rate_pct < 24:  return "Below Average",   "#e67e22"
    return "Higher Risk",  "#e74c3c"


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
    "🤖 Loan Predictor",
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

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 6 · LOAN PREDICTOR                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tabs[5]:
    st.markdown("### 🤖 Loan Predictor")
    insight(
        "This tool combines a <b>Gradient Boosting ML model</b> (for loan amount) with "
        "<b>historical percentile analysis</b> (for interest rate) across 270,000 real loans. "
        "The interest rate is driven primarily by credit score (FICO) — data not captured here — "
        "so instead of an unreliable regression, we show the actual rate range received by "
        "borrowers with your income and purpose profile. This is more honest and more useful."
    )

    # ── Train models (cached) ─────────────────────────────────────────────
    (pipe_amt, r2_amt,
     rate_lookup, rate_lookup_inc,
     bg, feat_names, importance) = train_models(df)

    st.markdown("#### Enter Your Profile")

    # ── Input dropdowns ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        sel_income = st.selectbox(
            "💵 Annual Income Range",
            list(INCOME_OPTIONS.keys()),
            index=2,
            help="Select the bracket closest to your gross annual income."
        )
    with c2:
        sel_emp = st.selectbox(
            "💼 Employment Length",
            list(EMP_OPTIONS.keys()),
            index=4,
            help="How many years you have been continuously employed."
        )
    with c3:
        purpose_labels = list(PURPOSE_DISPLAY.values())
        purpose_keys   = list(PURPOSE_DISPLAY.keys())
        sel_purpose_label = st.selectbox(
            "🎯 Loan Purpose",
            purpose_labels,
            index=0,
            help="What will you use the loan for?"
        )
        sel_purpose = purpose_keys[purpose_labels.index(sel_purpose_label)]
    with c4:
        sel_home_label = st.selectbox(
            "🏠 Home Ownership",
            [HOME_DISPLAY[h] for h in HOME_OPTIONS],
            index=1,
            help="Your current housing situation."
        )
        sel_home = HOME_OPTIONS[[HOME_DISPLAY[h] for h in HOME_OPTIONS].index(sel_home_label)]

    st.markdown("")
    predict_btn = st.button("🔍 Predict My Loan", type="primary")

    # ── Prediction ────────────────────────────────────────────────────────
    if predict_btn:
        income_val = INCOME_OPTIONS[sel_income]
        emp_yrs    = EMP_OPTIONS[sel_emp]

        # Background balance features (median for this income bucket)
        inc_cut = pd.cut([income_val], bins=INCOME_BINS, labels=INCOME_LABELS)[0]
        avg_bal = float(bg.loc[inc_cut, "avg_cur_bal"]) if inc_cut in bg.index else float(bg["avg_cur_bal"].median())
        tot_bal = float(bg.loc[inc_cut, "Tot_cur_bal"]) if inc_cut in bg.index else float(bg["Tot_cur_bal"].median())

        X_pred = pd.DataFrame([{
            "annual_inc":     income_val,
            "emp_length_yrs": emp_yrs,
            "avg_cur_bal":    avg_bal,
            "Tot_cur_bal":    tot_bal,
            "purpose":        sel_purpose,
            "home_ownership": sel_home,
        }])

        raw_amt  = float(pipe_amt.predict(X_pred)[0])
        pred_amt = max(1_000, min(40_000, round(raw_amt / 500) * 500))

        # Rate range from historical data
        rate_p25, rate_med, rate_p75 = get_rate_range(
            rate_lookup, rate_lookup_inc, inc_cut, sel_purpose
        )

        # Installments at median rate
        inst_36 = calc_installment(pred_amt, rate_med, 36)
        inst_60 = calc_installment(pred_amt, rate_med, 60)

        label, label_color = profile_label(rate_med)

        st.markdown("---")
        st.markdown("#### Your Estimated Loan Offer")

        # ── Result cards ─────────────────────────────────────────────────
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("💰 Loan Amount",         f"${pred_amt:,.0f}",
                  help="ML model prediction based on your profile.")
        r2.metric("📈 Typical Rate",        f"{rate_med:.1f}%",
                  help=f"Median rate for similar borrowers. Range: {rate_p25:.1f}% – {rate_p75:.1f}%")
        r3.metric("📉 Best Rate (25th %)",  f"{rate_p25:.1f}%",
                  help="25% of similar borrowers received this rate or lower.")
        r4.metric("🗓️ Monthly · 36 months",  f"${inst_36:,.0f}",
                  help="At median interest rate.")
        r5.metric("🗓️ Monthly · 60 months",  f"${inst_60:,.0f}",
                  help="At median interest rate.")

        # ── Profile banner ────────────────────────────────────────────────
        total_interest_36 = inst_36 * 36 - pred_amt
        total_interest_60 = inst_60 * 60 - pred_amt
        interest_saved    = total_interest_60 - total_interest_36
        st.markdown(
            f"<div style='background:{label_color}22; border-left:4px solid {label_color}; "
            f"border-radius:0 8px 8px 0; padding:14px 20px; margin:12px 0;'>"
            f"<span style='color:{label_color}; font-weight:700; font-size:1.05rem;'>"
            f"{label} Credit Profile</span><br>"
            f"<span style='color:#aabbcc; font-size:0.9rem;'>"
            f"Borrowers with your profile ({sel_income} income, {sel_emp} of employment, "
            f"<b>{sel_purpose_label}</b>, <b>{sel_home_label}</b>) historically received rates "
            f"between <b>{rate_p25:.1f}% – {rate_p75:.1f}%</b>, with a median of <b>{rate_med:.1f}%</b>. "
            f"The model estimates an approval of <b>${pred_amt:,.0f}</b>. "
            f"Choosing 36 months over 60 months saves approximately "
            f"<b>${interest_saved:,.0f}</b> in total interest."
            f"</span></div>",
            unsafe_allow_html=True,
        )


    else:
        st.markdown(
            "<div style='text-align:center; padding:40px; color:#445566; font-size:1rem;'>"
            "Fill in your profile above and click <b style='color:#00d4ff;'>Predict My Loan</b> "
            "to see your estimated loan offer."
            "</div>",
            unsafe_allow_html=True,
        )

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
