import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Portfolio Auto Analyzer", layout="wide")

# ------------------------------------------------------
# PDF PARSER (FULLY UPDATED FOR SLA FINSERV / MINT PDFs)
# ------------------------------------------------------
def extract_mf_table(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            for table in tables:
                df = pd.DataFrame(table)

                # Clean raw data
                df.replace("", None, inplace=True)
                df.dropna(how="all", axis=0, inplace=True)
                df.dropna(how="all", axis=1, inplace=True)

                # Skip tiny tables
                if df.shape[1] < 3:
                    continue

                # Detect MF table using headers
                header_text = " ".join(df.iloc[0].astype(str).tolist()).lower()

                if (
                    "scheme" in header_text
                    or "fund" in header_text
                    or "purchase value" in header_text
                    or "current value" in header_text
                    or "allocation" in header_text
                ):
                    # First row = header
                    df.columns = df.iloc[0]
                    df = df[1:]

                    # Clean column names
                    df.columns = (
                        df.columns.str.strip()
                        .str.replace("\n", " ")
                        .str.replace("  ", " ")
                    )

                    # Select relevant columns
                    keep = [c for c in df.columns if any(
                        k in str(c).lower() for k in [
                            "scheme", "fund", "sub", "category",
                            "purchase", "current", "allocation", "%"
                        ])]

                    df = df[keep]

                    # Drop empty rows
                    df.dropna(how="all", axis=0, inplace=True)

                    # Try converting numbers
                    for col in df.columns:
                        try:
                            df[col] = (
                                df[col].astype(str)
                                .str.replace(",", "")
                                .str.replace("%", "")
                                .astype(float)
                            )
                        except:
                            pass

                    return df

    return None

# -----------------------
# CATEGORY CLASSIFICATION
# -----------------------
def classify(row):
    sub = str(row).lower()
    if "liquid" in sub:
        return "Liquid"
    if "gilt" in sub or "debt" in sub:
        return "Debt"
    if "equity" in sub or "mid" in sub or "small" in sub or "large" in sub or "flexi" in sub:
        return "Equity"
    return "Other"

# -------------------------------
# TARGET ALLOCATION (3 MODELS)
# -------------------------------
def get_target_allocation(model):
    if model == "Conservative":
        return {"Equity": 30, "Debt": 40, "Liquid": 30}
    if model == "Moderate":
        return {"Equity": 50, "Debt": 25, "Liquid": 25}
    if model == "Aggressive":
        return {"Equity": 60, "Debt": 20, "Liquid": 20}

# -------------------------------
# AMC EXPOSURE (20% limit)
# -------------------------------
def amc_exposure_alerts(df):
    alerts = []
    if "Fund" in df.columns:
        amcs = df.groupby("Fund")["Allocation (%)"].sum()
        for amc, pct in amcs.items():
            if pct > 20:
                alerts.append(f"⚠ AMC Overweight: {amc} = {pct:.2f}% (Limit: 20%)")
    return alerts

# -------------------------------
# SUGGESTION ENGINE
# -------------------------------
def generate_suggestions(df, model):

    # Calculate category sums
    df["Category"] = df["Sub Category"].apply(classify)

    total = df["Current Value"].sum()
    eq = df[df["Category"] == "Equity"]["Current Value"].sum()
    debt = df[df["Category"] == "Debt"]["Current Value"].sum()
    liq = df[df["Category"] == "Liquid"]["Current Value"].sum()

    eq_pct = round(eq / total * 100, 2)
    debt_pct = round(debt / total * 100, 2)
    liq_pct = round(liq / total * 100, 2)

    target = get_target_allocation(model)

    suggestions = []

    # ---------------- SIP Suggestions ----------------
    if eq_pct < target["Equity"]:
        diff = target["Equity"] - eq_pct
        suggestions.append(f"📌 Increase SIP in existing equity SIP funds to reach +{diff:.2f}% equity.")
    else:
        suggestions.append("✔ Equity SIP allocation looks fine.")

    # ---------------- Lumpsum Suggestions ----------------
    if liq_pct > target["Liquid"]:
        suggestions.append("📌 Invest part of Liquid funds into Equity/Multi-cap based on selected risk model.")

    # ---------------- Redemption Suggestions ----------------
    suggestions.append("📌 For withdrawals: Redeem from Liquid → then Debt → avoid Equity.")

    # ---------------- AMC LIMIT ----------------
    amc_alert = amc_exposure_alerts(df)
    suggestions.extend(amc_alert)

    return suggestions, {"Equity": eq_pct, "Debt": debt_pct, "Liquid": liq_pct}

# ------------------------------------------------------
# FRONTEND UI
# ------------------------------------------------------
st.title("📊 Portfolio Auto Analyzer")
st.write("Upload your Mutual Fund portfolio PDF to get automatic analysis + suggestions.")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

risk_model = st.radio("Select Risk Profile", ["Conservative", "Moderate", "Aggressive"])

if uploaded_pdf:
    st.success("PDF uploaded! Reading data...")

    df = extract_mf_table(uploaded_pdf)

    if df is None:
        st.error("❌ Unable to detect Mutual Fund table. PDF format mismatch.")
    else:
        st.subheader("📄 Extracted Mutual Fund Data")
        st.dataframe(df)

        # Suggestions
        st.subheader("📌 Suggestions")
        s, alloc = generate_suggestions(df, risk_model)

        for x in s:
            st.write(x)

        # Allocation chart
        st.subheader("📊 Category Allocation")

        fig, ax = plt.subplots(facecolor="#111111")
        ax.set_facecolor("#111111")

        labels = list(alloc.keys())
        values = list(alloc.values())

        ax.bar(labels, values)
        ax.set_title("Category Allocation", color="white")
        ax.tick_params(colors="white")

        st.pyplot(fig)
