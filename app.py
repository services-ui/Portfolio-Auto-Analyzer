import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Auto Analyzer", layout="wide")

# ----------------- Helpers -----------------
def clean_df(raw_table):
    df = pd.DataFrame(raw_table)
    df.replace("", None, inplace=True)
    df.dropna(how="all", axis=0, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    if df.empty or df.shape[0] < 2:
        return None
    header = df.iloc[0]
    df = df[1:].copy()
    df.columns = header
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ")
        .str.replace("  ", " ")
    )
    return df


def num(series):
    try:
        return (
            series.astype(str)
            .str.replace(",", "")
            .str.replace("%", "")
            .str.strip()
            .replace("", "0")
            .astype(float)
        )
    except Exception:
        return series


# ----------------- PDF Parsing -----------------
def parse_pdf_sections(pdf_file):
    """
    Returns a dict:
    {
      'applicant': df,
      'fund': df,
      'scheme': df,
      'subcat': df,
      'sip': df
    }
    """
    sections = {
        "applicant": None,
        "fund": None,
        "scheme": None,
        "subcat": None,
        "sip": None,
    }

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables:
                df = clean_df(t)
                if df is None:
                    continue

                headers_lower = " ".join(df.columns.astype(str).str.lower().tolist())

                # Applicant summary: Applicant, Purchase Value, Current Value, CAGR
                if (
                    "applicant" in headers_lower
                    and "purchase value" in headers_lower
                    and "current value" in headers_lower
                    and sections["applicant"] is None
                ):
                    sections["applicant"] = df.copy()
                    continue

                # Fund-wise allocation
                if (
                    "fund" in headers_lower
                    and "allocation" in headers_lower
                    and "purchase value" in headers_lower
                    and sections["fund"] is None
                ):
                    sections["fund"] = df.copy()
                    continue

                # Scheme-wise allocation
                if (
                    "scheme" in headers_lower
                    and "allocation" in headers_lower
                    and "purchase value" in headers_lower
                    and sections["scheme"] is None
                ):
                    sections["scheme"] = df.copy()
                    continue

                # Sub category allocation
                if (
                    "sub category" in headers_lower
                    and "allocation" in headers_lower
                    and sections["subcat"] is None
                ):
                    sections["subcat"] = df.copy()
                    continue

                # SIP Summary
                if (
                    "sip summary" in headers_lower
                    or ("folio" in headers_lower and "scheme" in headers_lower)
                ) and sections["sip"] is None:
                    sections["sip"] = df.copy()
                    continue

    return sections


# ----------------- Allocation logic -----------------
def classify_subcat(subcat):
    s = str(subcat).lower()
    if "liquid" in s:
        return "Liquid"
    if "gilt" in s or "debt" in s:
        return "Debt"
    if "equity" in s or "mid" in s or "small" in s or "flexi" in s or "large" in s:
        return "Equity"
    return "Other"


def get_target_allocation(model):
    if model == "Conservative":
        return {"Equity": 30, "Debt": 40, "Liquid": 30}
    if model == "Moderate":
        return {"Equity": 50, "Debt": 25, "Liquid": 25}
    if model == "Aggressive":
        return {"Equity": 60, "Debt": 20, "Liquid": 20}


def compute_current_allocation(subcat_df):
    if subcat_df is None:
        return None
    df = subcat_df.copy()
    if "Allocation (%)" in df.columns:
        df["Allocation (%)"] = num(df["Allocation (%)"])
    df["Bucket"] = df["Sub Category"].apply(classify_subcat)
    alloc = df.groupby("Bucket")["Allocation (%)"].sum().to_dict()
    for k in ["Equity", "Debt", "Liquid"]:
        alloc.setdefault(k, 0.0)
    return alloc


def amc_exposure(fund_df):
    alerts = []
    if fund_df is None:
        return alerts
    df = fund_df.copy()
    if "Allocation (%)" in df.columns:
        df["Allocation (%)"] = num(df["Allocation (%)"])
        grouped = df.groupby("Fund")["Allocation (%)"].sum()
        for amc, pct in grouped.items():
            if pct > 20:
                alerts.append(f"⚠ {amc} = {pct:.2f}% (> 20% AMC limit)")
    return alerts


# ----------------- Action suggestions -----------------
def suggest_increase_sip(sip_df, extra_amount):
    if sip_df is None:
        return "No SIP data found in PDF."

    df = sip_df.copy()
    # First column should be Scheme name
    scheme_col = [c for c in df.columns if "scheme" in str(c).lower()]
    if not scheme_col:
        return "Could not detect Scheme column in SIP summary."
    scheme_col = scheme_col[0]

    schemes = df[scheme_col].dropna().unique().tolist()
    if not schemes:
        return "No active SIP schemes found."

    per_scheme = extra_amount / len(schemes) if extra_amount > 0 else 0
    res = pd.DataFrame(
        {"Scheme": schemes, "Additional SIP / month (₹)": [round(per_scheme, 2)] * len(schemes)}
    )
    return res


def suggest_lumpsum(scheme_df, subcat_df, amount, model):
    if scheme_df is None or subcat_df is None or amount <= 0:
        return "Not enough data to suggest lumpsum allocation."

    alloc = compute_current_allocation(subcat_df)
    target = get_target_allocation(model)

    equity_gap = target["Equity"] - alloc.get("Equity", 0)
    if equity_gap <= 0:
        note = "Current equity % is already at or above target. You may invest lumpsum into Debt or Liquid as per need."
        return note

    # identify equity-oriented schemes (simple name-based)
    df = scheme_df.copy()
    scheme_col = [c for c in df.columns if "scheme" in str(c).lower()][0]
    curr_val_col = [c for c in df.columns if "current value" in str(c).lower()][0]

    df[curr_val_col] = num(df[curr_val_col])

    def is_equity_name(x):
        s = str(x).lower()
        return any(k in s for k in ["flexi", "mid", "small", "equity"])

    eq_df = df[df[scheme_col].apply(is_equity_name)].copy()
    if eq_df.empty:
        return "No equity schemes found to invest lumpsum."

    total_curr = eq_df[curr_val_col].sum()
    eq_df["Suggested Lumpsum (₹)"] = eq_df[curr_val_col] / total_curr * amount
    return eq_df[[scheme_col, "Suggested Lumpsum (₹)"]]


def suggest_redeem(scheme_df, amount):
    if scheme_df is None or amount <= 0:
        return "Not enough data to suggest redemption."

    df = scheme_df.copy()
    scheme_col = [c for c in df.columns if "scheme" in str(c).lower()][0]
    curr_val_col = [c for c in df.columns if "current value" in str(c).lower()][0]
    df[curr_val_col] = num(df[curr_val_col])

    def is_liquid_name(x):
        return "liquid" in str(x).lower()

    liq_df = df[df[scheme_col].apply(is_liquid_name)].copy()
    if liq_df.empty:
        return "No Liquid schemes found – you may need to redeem from Debt funds manually."

    total_liq = liq_df[curr_val_col].sum()
    if total_liq == 0:
        return "Liquid value is zero."

    liq_df["Suggested Redemption (₹)"] = liq_df[curr_val_col] / total_liq * amount
    return liq_df[[scheme_col, "Suggested Redemption (₹)"]]


# ----------------- UI -----------------
st.title("📊 Portfolio Auto Analyzer")
st.write("Upload your Mutual Fund portfolio PDF (same SLA Finserv format) to see summary and get suggestions.")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
risk_model = st.radio("Select Risk Profile", ["Conservative", "Moderate", "Aggressive"])

if uploaded:
    sections = parse_pdf_sections(uploaded)

    applicant = sections["applicant"]
    fund = sections["fund"]
    scheme = sections["scheme"]
    subcat = sections["subcat"]
    sip = sections["sip"]

    # ---- 1. Applicant Summary ----
    st.header("1️⃣ Portfolio Summary")

    if applicant is not None:
        a = applicant.copy()
        a_cols = [c.lower() for c in a.columns]

        # Try to pull values from first row
        row = a.iloc[0]
        name = row.get("Applicant", "N/A")
        pur = row.get("Purchase Value", None)
        cur = row.get("Current Value", None)
        abs_ret = row.get("Absolute Return (%)", None)
        cagr = row.get("CAGR (%)", None)

        col1, col2, col3 = st.columns(3)
        col1.metric("Client Name", str(name))
        if pur is not None:
            col2.metric("Purchase Value (₹)", f"{pur}")
        if cur is not None:
            col3.metric("Current Value (₹)", f"{cur}")

        col4, col5 = st.columns(2)
        if abs_ret is not None:
            col4.metric("Absolute Return (%)", f"{abs_ret}")
        if cagr is not None:
            col5.metric("CAGR (%)", f"{cagr}")

        st.write("Raw table:")
        st.dataframe(applicant)
    else:
        st.info("Applicant summary table not detected.")

    # ---- 2. AMC Wise Allocation ----
    st.header("2️⃣ AMC-wise Allocation")
    if fund is not None:
        # numeric clean
        if "Allocation (%)" in fund.columns:
            fund["Allocation (%)"] = num(fund["Allocation (%)"])
        st.dataframe(fund)

        fig1, ax1 = plt.subplots()
        ax1.bar(fund["Fund"].astype(str), fund["Allocation (%)"])
        ax1.set_xticklabels(fund["Fund"].astype(str), rotation=45, ha="right")
        ax1.set_ylabel("Allocation (%)")
        st.pyplot(fig1)
    else:
        st.info("AMC allocation table not detected.")

    # ---- 3. Sub Category Allocation ----
    st.header("3️⃣ Sub-category Allocation")
    if subcat is not None:
        if "Allocation (%)" in subcat.columns:
            subcat["Allocation (%)"] = num(subcat["Allocation (%)"])
        st.dataframe(subcat)

        alloc = compute_current_allocation(subcat)
        if alloc:
            fig2, ax2 = plt.subplots()
            ax2.bar(list(alloc.keys()), list(alloc.values()))
            ax2.set_ylabel("Allocation (%)")
            st.pyplot(fig2)
    else:
        st.info("Sub-category allocation table not detected.")

    # ---- 4. SIP Summary ----
    st.header("4️⃣ SIP Summary")
    if sip is not None:
        st.dataframe(sip)
    else:
        st.info("SIP summary table not detected.")

    # ---- AMC exposure alerts ----
    st.subheader("⚠ AMC Exposure Check")
    alerts = amc_exposure(fund)
    if alerts:
        for a in alerts:
            st.write(a)
    else:
        st.write("No AMC above 20% (or data not available).")

    # ---- What do you want to do? ----
    st.header("5️⃣ Action & Suggestions")

    action = st.radio("What do you want to do?", ["Increase SIP", "Invest Lumpsum", "Redeem"])
    amount = st.number_input("Enter amount (₹)", min_value=0.0, step=1000.0)

    if st.button("Show Suggestion"):
        if action == "Increase SIP":
            res = suggest_increase_sip(sip, amount)
            if isinstance(res, pd.DataFrame):
                st.write("Suggested increase in SIP per scheme:")
                st.dataframe(res)
            else:
                st.write(res)

        elif action == "Invest Lumpsum":
            res = suggest_lumpsum(scheme, subcat, amount, risk_model)
            if isinstance(res, pd.DataFrame):
                st.write("Suggested lumpsum allocation into schemes:")
                st.dataframe(res)
            else:
                st.write(res)

        elif action == "Redeem":
            res = suggest_redeem(scheme, amount)
            if isinstance(res, pd.DataFrame):
                st.write("Suggested redemption from Liquid schemes:")
                st.dataframe(res)
            else:
                st.write(res)
