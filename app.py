import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Auto Analyzer", layout="wide")


# ---------- Helper functions ----------

def num(series):
    """Convert a column to numeric if possible."""
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


def find_col(df, keywords):
    """Find first column whose name contains any of the given keywords."""
    cols = df.columns.astype(str)
    for k in keywords:
        for c in cols:
            if k.lower() in c.lower():
                return c
    return None


def classify_subcat(text):
    s = str(text).lower()
    if "liquid" in s:
        return "Liquid"
    if "gilt" in s or "debt" in s:
        return "Debt"
    if any(x in s for x in ["equity", "mid", "small", "large", "flexi"]):
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
    alloc_col = find_col(df, ["allocation"])
    subcat_col = find_col(df, ["sub category", "subcategory"])
    if alloc_col is None or subcat_col is None:
        return None
    df[alloc_col] = num(df[alloc_col])
    df["Bucket"] = df[subcat_col].apply(classify_subcat)
    res = df.groupby("Bucket")[alloc_col].sum().to_dict()
    for k in ["Equity", "Debt", "Liquid"]:
        res.setdefault(k, 0.0)
    return res


def amc_exposure(fund_df):
    alerts = []
    if fund_df is None:
        return alerts
    df = fund_df.copy()
    alloc_col = find_col(df, ["allocation"])
    fund_col = find_col(df, ["fund"])
    if alloc_col is None or fund_col is None:
        return alerts
    df[alloc_col] = num(df[alloc_col])
    grouped = df.groupby(fund_col)[alloc_col].sum()
    for amc, pct in grouped.items():
        if pct > 20:
            alerts.append(f"⚠ {amc} = {pct:.2f}% (> 20% AMC limit)")
    return alerts


def suggest_increase_sip(sip_df, amount):
    if sip_df is None or amount <= 0:
        return "No SIP data or amount = 0."

    scheme_col = find_col(sip_df, ["scheme"])
    if scheme_col is None:
        return "Could not detect Scheme column in SIP sheet."

    schemes = sip_df[scheme_col].dropna().unique().tolist()
    if not schemes:
        return "No SIP schemes found."

    per_scheme = amount / len(schemes)
    res = pd.DataFrame(
        {"Scheme": schemes, "Additional SIP / month (₹)": [round(per_scheme, 2)] * len(schemes)}
    )
    return res


def suggest_lumpsum(scheme_df, subcat_df, amount, model):
    if scheme_df is None or subcat_df is None or amount <= 0:
        return "Not enough data or amount = 0."

    target = get_target_allocation(model)
    alloc = compute_current_allocation(subcat_df)
    if alloc is None:
        return "Sub-category allocation not clear."

    equity_gap = target["Equity"] - alloc.get("Equity", 0)
    if equity_gap <= 0:
        return "Equity already at/above target %. Consider Debt/Liquid."

    sch = scheme_df.copy()
    scheme_col = find_col(sch, ["scheme"])
    curr_col = find_col(sch, ["current value"])
    if scheme_col is None or curr_col is None:
        return "Could not detect Scheme / Current Value columns."

    sch[curr_col] = num(sch[curr_col])

    def is_equity_name(x):
        s = str(x).lower()
        return any(k in s for k in ["flexi", "mid", "small", "equity", "large"])

    eq_df = sch[sch[scheme_col].apply(is_equity_name)].copy()
    if eq_df.empty:
        return "No equity schemes found for lumpsum."

    total_curr = eq_df[curr_col].sum()
    if total_curr == 0:
        return "Equity current value is zero."

    eq_df["Suggested Lumpsum (₹)"] = eq_df[curr_col] / total_curr * amount
    return eq_df[[scheme_col, "Suggested Lumpsum (₹)"]]


def suggest_redeem(scheme_df, amount):
    if scheme_df is None or amount <= 0:
        return "Not enough data or amount = 0."

    sch = scheme_df.copy()
    scheme_col = find_col(sch, ["scheme"])
    curr_col = find_col(sch, ["current value"])
    if scheme_col is None or curr_col is None:
        return "Could not detect Scheme / Current Value columns."

    sch[curr_col] = num(sch[curr_col])

    def is_liquid_name(x):
        return "liquid" in str(x).lower()

    liq_df = sch[sch[scheme_col].apply(is_liquid_name)].copy()
    if liq_df.empty:
        return "No Liquid schemes found – redeem from Debt manually."

    total_liq = liq_df[curr_col].sum()
    if total_liq == 0:
        return "Liquid value is zero."

    liq_df["Suggested Redemption (₹)"] = liq_df[curr_col] / total_liq * amount
    return liq_df[[scheme_col, "Suggested Redemption (₹)"]]


# ---------- UI ----------

st.title("📊 Portfolio Auto Analyzer (Excel Version)")
st.write("Upload your Mutual Fund portfolio **Excel file** (same format each time).")

uploaded = st.file_uploader("Upload Excel", type=["xlsx", "xls"])
risk_model = st.radio("Select Risk Profile", ["Conservative", "Moderate", "Aggressive"])

if uploaded:
    st.success("Excel uploaded – reading sheets...")

    # read all sheets into dict: {sheet_name: dataframe}
    try:
        sheets = pd.read_excel(uploaded, sheet_name=None)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        st.stop()

    sheet_names = list(sheets.keys())
    st.info(f"Detected {len(sheet_names)} sheets: {', '.join(sheet_names)}")

    # show previews
    for name in sheet_names:
        with st.expander(f"Preview sheet: {name}"):
            st.dataframe(sheets[name])

    st.markdown("---")
    st.subheader("Step 1 – Map sheets")

    sheet_summary = st.selectbox(
        "Sheet for Portfolio Summary (Name, Purchase, Current, Returns)",
        options=sheet_names,
    )

    sheet_amc = st.selectbox(
        "Sheet for AMC-wise Allocation",
        options=sheet_names,
    )

    sheet_subcat = st.selectbox(
        "Sheet for Sub-category Allocation",
        options=sheet_names,
    )

    sheet_scheme = st.selectbox(
        "Sheet for Scheme-wise Allocation (Purchase/Current Value)",
        options=sheet_names,
    )

    sheet_sip = st.selectbox(
        "Sheet for SIP Summary",
        options=sheet_names,
    )

    summary_df = sheets[sheet_summary].copy()
    amc_df = sheets[sheet_amc].copy()
    subcat_df = sheets[sheet_subcat].copy()
    scheme_df = sheets[sheet_scheme].copy()
    sip_df = sheets[sheet_sip].copy()

    # ---------- 1. Portfolio summary ----------
    st.markdown("---")
    st.header("1️⃣ Portfolio Summary")

    name_col = find_col(summary_df, ["applicant", "name"])
    pur_col = find_col(summary_df, ["purchase value"])
    cur_col = find_col(summary_df, ["current value"])
    abs_col = find_col(summary_df, ["absolute"])
    cagr_col = find_col(summary_df, ["cagr"])

    name_val = summary_df[name_col].iloc[0] if name_col else "N/A"
    col1, col2, col3 = st.columns(3)
    col1.metric("Client Name", str(name_val))
    if pur_col:
        col2.metric("Purchase Value (₹)", str(summary_df[pur_col].iloc[0]))
    if cur_col:
        col3.metric("Current Value (₹)", str(summary_df[cur_col].iloc[0]))

    col4, col5 = st.columns(2)
    if abs_col:
        col4.metric("Absolute Return (%)", str(summary_df[abs_col].iloc[0]))
    if cagr_col:
        col5.metric("CAGR (%)", str(summary_df[cagr_col].iloc[0]))

    st.write("Summary sheet:")
    st.dataframe(summary_df)

    # ---------- 2. AMC-wise allocation ----------
    st.header("2️⃣ AMC-wise Allocation")
    st.dataframe(amc_df)

    alloc_col_amc = find_col(amc_df, ["allocation"])
    fund_col_amc = find_col(amc_df, ["fund"])
    if alloc_col_amc and fund_col_amc:
        amc_df[alloc_col_amc] = num(amc_df[alloc_col_amc])
        fig1, ax1 = plt.subplots()
        ax1.bar(amc_df[fund_col_amc].astype(str), amc_df[alloc_col_amc])
        ax1.set_xticklabels(amc_df[fund_col_amc].astype(str), rotation=45, ha="right")
        ax1.set_ylabel("Allocation (%)")
        st.pyplot(fig1)

    st.subheader("⚠ AMC Exposure Check (20% limit)")
    alerts = amc_exposure(amc_df)
    if alerts:
        for a in alerts:
            st.write(a)
    else:
        st.write("No AMC above 20% (or allocation data not clear).")

    # ---------- 3. Sub-category allocation ----------
    st.header("3️⃣ Sub-category Allocation")
    st.dataframe(subcat_df)

    alloc = compute_current_allocation(subcat_df)
    if alloc:
        fig2, ax2 = plt.subplots()
        ax2.bar(list(alloc.keys()), list(alloc.values()))
        ax2.set_ylabel("Allocation (%)")
        st.pyplot(fig2)

    # ---------- 4. SIP Summary ----------
    st.header("4️⃣ SIP Summary")
    st.dataframe(sip_df)

    # ---------- 5. Action & Suggestions ----------
    st.markdown("---")
    st.header("4️⃣ Action & Suggestions")

    action = st.radio("What do you want to do?", ["Increase SIP", "Invest Lumpsum", "Redeem"])
    amt = st.number_input("Enter amount (₹)", min_value=0.0, step=1000.0)

    if st.button("Show Suggestion"):
        if action == "Increase SIP":
            res = suggest_increase_sip(sip_df, amt)
            if isinstance(res, pd.DataFrame):
                st.write("Suggested extra SIP per scheme:")
                st.dataframe(res)
            else:
                st.write(res)

        elif action == "Invest Lumpsum":
            res = suggest_lumpsum(scheme_df, subcat_df, amt, risk_model)
            if isinstance(res, pd.DataFrame):
                st.write("Suggested lumpsum allocation:")
                st.dataframe(res)
            else:
                st.write(res)

        elif action == "Redeem":
            res = suggest_redeem(scheme_df, amt)
            if isinstance(res, pd.DataFrame):
                st.write("Suggested redemption from Liquid schemes:")
                st.dataframe(res)
            else:
                st.write(res)
