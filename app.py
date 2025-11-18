import streamlit as st
import pandas as pd
import pdfplumber
import re
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Auto Analyzer", layout="wide")

# -------------------------------
# PDF PARSER (your SLA Mint format)
# -------------------------------
def extract_tables_from_pdf(upload):
    data = []
    with pdfplumber.open(upload) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table)
                data.append(df)
    return data

# -------------------------------
# CLEAN TABLE FOR MF ONLY
# -------------------------------
def get_mutual_funds(clean_tables):
    mf_rows = []
    for df in clean_tables:
        for i, row in df.iterrows():
            if "Fund" in str(row[0]) or "Scheme" in str(row[0]):
                continue
            # Key pattern for MF row: contains NAV or Allocation %
            if re.search(r'\d', str(row[0])):
                continue
        # Try identifying scheme table by known column patterns
        if df.shape[1] >= 5:
            if "Purchase Value" in df.iloc[0].astype(str).tolist():
                df.columns = df.iloc[0]
                df = df[1:]
                return df
    return None

# -------------------------------
# CATEGORY DETECTION
# -------------------------------
def categorize(subcat):
    if "Liquid" in subcat:
        return "Liquid"
    if "Gilt" in subcat or "Debt" in subcat:
        return "Debt"
    if "Equity" in subcat:
        return "Equity"
    return "Others"

# -------------------------------
# AMC Exposure Check
# -------------------------------
def check_amc_exposure(df):
    alerts = []
    grouped = df.groupby("Fund")["Allocation (%)"].sum()
    for amc, val in grouped.items():
        if float(val) > 20:
            alerts.append(f"⚠ {amc} AMC is {val}% (Above 20% limit)")
    return alerts

# -------------------------------
# RISK MODEL TARGET ALLOCATION
# -------------------------------
def get_target_allocation(model):
    if model == "Conservative":
        return {"Equity": 30, "Debt": 40, "Liquid": 30}
    if model == "Moderate":
        return {"Equity": 50, "Debt": 25, "Liquid": 25}
    if model == "Aggressive":
        return {"Equity": 60, "Debt": 20, "Liquid": 20}

# -------------------------------
# SUGGESTION ENGINE
# -------------------------------
def suggestions(df, model):

    # Extract needed data
    total = df["Current Value"].astype(float).sum()
    equity = df[df["Sub Category"].str.contains("Equity")]["Current Value"].astype(float).sum()
    debt = df[df["Sub Category"].str.contains("Debt")]["Current Value"].astype(float).sum()
    liquid = df[df["Sub Category"].str.contains("Liquid")]["Current Value"].astype(float).sum()

    eq_pct = round((equity / total) * 100, 2)
    dt_pct = round((debt / total) * 100, 2)
    liq_pct = round((liquid / total) * 100, 2)

    target = get_target_allocation(model)

    sug = []

    # ---------------- SIP SUGGESTION ----------------
    if eq_pct < target["Equity"]:
        diff = target["Equity"] - eq_pct
        sug.append(f"👉 Increase SIP in existing Equity SIP funds to reach +{diff:.2f}% equity.")
    else:
        sug.append("✔ Equity SIP allocation is aligned.")

    # ---------------- LUMPSUM SUGGESTION ----------------
    if liq_pct > target["Liquid"]:
        sug.append("👉 Shift portion of Liquid funds into Equity/Multi-cap fund (based on model).")

    # ---------------- REDEMPTION SUGGESTION ----------------
    if liq_pct < 10:
        sug.append("✔ Liquid allocation is low. Avoid redemption.")
    else:
        sug.append("👉 If you need cash, redeem from Liquid funds first (your rule).")

    # ---------------- AMC EXPOSURE ----------------
    amc_alerts = check_amc_exposure(df)
    sug.extend(amc_alerts)

    return sug, {"Equity": eq_pct, "Debt": dt_pct, "Liquid": liq_pct}

# -------------------------------
# FRONTEND UI
# -------------------------------
st.title("📊 Portfolio Auto Analyzer")

st.write("Upload your Mutual Fund portfolio PDF to get automatic analysis + suggestions.")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

risk_model = st.radio("Select Risk Profile", ["Conservative", "Moderate", "Aggressive"])

if uploaded_pdf:
    st.success("PDF uploaded successfully! Reading data...")

    tables = extract_tables_from_pdf(uploaded_pdf)

    mf_table = get_mutual_funds(tables)

    if mf_table is None:
        st.error("Unable to detect MF table. PDF format mismatch.")
    else:
        st.subheader("Extracted Mutual Fund Table")
        st.dataframe(mf_table)

        # Suggestions
        st.subheader("📌 Suggestions")
        sug, alloc = suggestions(mf_table, risk_model)

        for s in sug:
            st.write(s)

        # Allocation Chart
        st.subheader("📊 Category Allocation")

        fig, ax = plt.subplots(facecolor="#111111")
        ax.set_facecolor("#111111")

        labels = list(alloc.keys())
        values = list(alloc.values())

        plt.bar(labels, values)
        plt.title("Category Allocation", color="white")
        plt.xticks(color="white")
        plt.yticks(color="white")

        st.pyplot(fig)
