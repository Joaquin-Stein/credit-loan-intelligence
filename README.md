# 💳 Credit Loan Intelligence Dashboard

An interactive data dashboard analyzing **270,000 US loan records (2012–2019)**, built with Python, Streamlit, and Plotly. Includes a machine learning loan predictor powered by Gradient Boosting.

🔗 **[Live Dashboard](https://credit-loan-intelligence-9tf4ypx4eadpop6mmqcqev.streamlit.app/)**

---

## Overview

FinTech is an industry where data drives every decision — from approving a loan to setting an interest rate. This project takes a real-world Kaggle dataset of over 270,000 US loans and turns it into an actionable intelligence dashboard, exploring borrower behavior, regional patterns, risk classification, and predictive modeling.

The dashboard is designed for data analysts, recruiters, and anyone curious about how lending data works in practice.

---

## Features

### 🏠 Overview
- Portfolio-level KPIs: total loans, total volume ($4.2B), average loan amount, average interest rate
- Loan status breakdown (Current, Fully Paid, Charged Off, Other)
- Risk category distribution (Not Risky / Risky / Very Risky)

### 👥 Customer Segmentation
- K-Means clustering (k=3) on annual income → Low / Mid / High Income segments
- Income distribution box plots and segment composition
- Loan behaviour by segment: average loan size, default rate, interest rate, loan purpose, home ownership

### 📊 Loan Analysis
- Risk classification by loan purpose (grouped bar: Risky + Very Risky only)
- Average loan amount by income segment and purpose
- Interest rate distribution across income segments

### 🗺️ Regional Analysis
- Choropleth map of loan volume and average loan amount across all 50 US states
- State-level breakdown table

### 📈 Temporal Trends
- Loan issuance trends from 2012 to 2019
- Interest rate evolution over time
- Risk trend by year

### 🤖 Loan Predictor (ML)
Enter a borrower profile and get an estimated loan offer:
- **Loan Amount** — predicted via Gradient Boosting Regressor trained on 270K records
- **Interest Rate Range** — derived from historical percentile analysis (p25 / median / p75) by income and purpose
- **Monthly Installments** — calculated for both 36-month and 60-month terms
- **Credit Profile** — color-coded tier based on predicted rate

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Dashboard framework |
| Plotly Express / Graph Objects | Interactive charts and choropleth map |
| pandas / numpy | Data cleaning and aggregation |
| scikit-learn | K-Means clustering, Gradient Boosting, Pipeline |

---

## Dataset

- **Source:** [Kaggle — Lending Club Loan Data](https://www.kaggle.com/)
- **Records:** 270,299 loans across 6 CSV files
- **Period:** 2012–2019
- **Geography:** United States

---

## Project Structure

```
loan-analysis/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Theme and app settings
└── data set/
    ├── loan_data_2012_2013.csv
    ├── loan_data_2014.csv
    ├── loan_data_2015.csv
    ├── loan_data_2016.csv
    ├── loan_data_2017_2019.csv
    └── loan_data_dict.csv
```

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/Joaquin-Stein/credit-loan-intelligence.git
cd credit-loan-intelligence

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## Key Insights

- **7.8%** of loans fall into the Very Risky category (charged off or defaulted)
- **Small business** loans carry the highest default risk by purpose
- **High Income** borrowers take larger loans but at significantly lower interest rates
- Loan volume grew steadily through 2015, then plateaued — reflecting tighter underwriting standards post-2016
- The ML predictor surfaces that **income range and loan purpose** are the strongest drivers of loan amount

---

*Built with Streamlit & deployed on Streamlit Community Cloud*
