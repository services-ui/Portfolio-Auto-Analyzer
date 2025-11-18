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


def classify_main_category(cat_text, subcat_text):
    s = (str(cat_text) + " " + str(subcat_text)).lower()
    if "liquid" in s:
        return "Liquid"
    if "gilt" in s or "debt" in s or "income" in s or "bond" in s:
        return "Debt"
    if any(x in s for x in ["equity", "mid", "small", "large", "flexi", "hybrid"]):
        return "Equity"
    return "Other"


def classify_sub_category(subcat_text):
    s = str(subcat_text).lower()
    if "large" in s:
        return "Large Cap"
    if "mid" in s:
        return "Mid Cap"
    if "small" in s:
        return "Small Cap"
    if "flexi" in s or "multi" in s:
        return "Flexi / Multi Cap"
    return "Others"


# ---------- UI ----------

st.title("📊 Portfolio Auto Analyzer (Excel)")
st.write("Upload your Mutual Fund portfolio **Excel file** (same SLA / Investwell format each time).")

uploaded = st.file_uploader("Upload Excel", type=["xlsx", "xls"])

if uploaded:
    st.success("Excel uploaded – reading data...")

    # Read FIRST sheet only (as you requested)
    try:
        df = pd.read_excel(uploaded, sheet_name=0)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        st.stop()

    # Show raw sheet preview
    with st.expander("Preview raw data"):
        st.dataframe(df)

    # Try to detect important columns
    name_col = find_col(df, ["applicant", "client name", "name"])
    cat_col = find_col(df, ["category"])
    subcat_col = find_col(df, ["sub category", "subcategory"])
    amc_col = find_col(df, ["fund", "amc", "fund house"])
    purch_col = find_col(df, ["purchase value", "inv amt"])
    curr_col = find_col(df, ["current value", "value"])

    # Clean numeric cols
    if purch_col:
        df[purch_col] = num(df[purch_col])
    if curr_col:
        df[curr_col] = num(df[curr_col])

    # ---------- 1. Name + Grand Totals ----------
    st.markdown("---")
    st.header("1️⃣ Portfolio Summary")

    # Name – take first non-empty value from name column, if available
    if name_col:
        name_val = df[name_col].dropna().astype(str).iloc[0]
    else:
        name_val = "N/A"

    total_purchase = df[purch_col].sum() if purch_col else 0
    total_current = df[curr_col].sum() if curr_col else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Client Name", str(name_val))
    c2.metric("Total Purchase Value (₹)", f"{total_purchase:,.0f}")
    c3.metric("Total Current Value (₹)", f"{total_current:,.0f}")

    # ---------- 2. Main Category Allocation (Equity / Debt / Liquid / Other) ----------
    st.markdown("---")
    st.header("2️⃣ Allocation by Main Category")

    if curr_col and (cat_col or subcat_col):
        df["MainCategory"] = df.apply(
            lambda r: classify_main_category(
                r[cat_col] if cat_col else "",
                r[subcat_col] if subcat_col else "",
            ),
            axis=1,
        )
        cat_alloc = df.groupby("MainCategory")[curr_col].sum()
        cat_alloc_pct = (cat_alloc / total_current * 100).round(2)

        alloc_table = pd.DataFrame(
            {
                "Current Value (₹)": cat_alloc,
                "Allocation (%)": cat_alloc_pct,
            }
        ).reset_index().rename(columns={"MainCategory": "Category"})

        st.dataframe(alloc_table)

        fig, ax = plt.subplots()
        ax.pie(cat_alloc_pct, labels=cat_alloc_pct.index, autopct="%1.1f%%")
        ax.set_title("Equity / Debt / Liquid / Other Allocation")
        st.pyplot(fig)
    else:
        st.info("Could not detect category columns to compute main allocation.")

    # ---------- 3. AMC-wise Allocation ----------
    st.markdown("---")
    st.header("3️⃣ AMC-wise Allocation")

    if curr_col and amc_col:
        amc_alloc_value = df.groupby(amc_col)[curr_col].sum()
        amc_alloc_pct = (amc_alloc_value / total_current * 100).round(2)

        amc_table = pd.DataFrame(
            {
                "AMC": amc_alloc_value.index.astype(str),
                "Current Value (₹)": amc_alloc_value.values,
                "Allocation (%)": amc_alloc_pct.values,
            }
        ).sort_values("Allocation (%)", ascending=False)

        st.dataframe(amc_table)

        fig2, ax2 = plt.subplots()
        ax2.bar(amc_table["AMC"], amc_table["Allocation (%)"])
        ax2.set_xticklabels(amc_table["AMC"], rotation=45, ha="right")
        ax2.set_ylabel("Allocation (%)")
        ax2.set_title("AMC-wise Allocation")
        st.pyplot(fig2)

        # 20% AMC rule alerts
        st.subheader("⚠ AMC > 20% Alerts")
        alerts = amc_table[amc_table["Allocation (%)"] > 20]
        if not alerts.empty:
            for _, row in alerts.iterrows():
                st.write(f"⚠ {row['AMC']} = {row['Allocation (%)']:.2f}% (> 20% limit)")
        else:
            st.write("No AMC above 20%.")
    else:
        st.info("Could not detect AMC / Current Value columns.")

    # ---------- 4. Sub-category Allocation (Large / Mid / Small / Flexi / Others) ----------
    st.markdown("---")
    st.header("4️⃣ Sub-category Allocation (Large/Mid/Small/Flexi/Others)")

    if curr_col and subcat_col:
        df["SubBucket"] = df[subcat_col].apply(classify_sub_category)
        sub_alloc_val = df.groupby("SubBucket")[curr_col].sum()
        sub_alloc_pct = (sub_alloc_val / total_current * 100).round(2)

        sub_table = pd.DataFrame(
            {
                "Sub-Category Bucket": sub_alloc_val.index.astype(str),
                "Current Value (₹)": sub_alloc_val.values,
                "Allocation (%)": sub_alloc_pct.values,
            }
        ).sort_values("Allocation (%)", ascending=False)

        st.dataframe(sub_table)

        fig3, ax3 = plt.subplots()
        ax3.bar(sub_table["Sub-Category Bucket"], sub_table["Allocation (%)"])
        ax3.set_xticklabels(sub_table["Sub-Category Bucket"], rotation=45, ha="right")
        ax3.set_ylabel("Allocation (%)")
        ax3.set_title("Sub-category Allocation")
        st.pyplot(fig3)
    else:
        st.info("Could not detect Sub Category column.")
else:
    st.info("Please upload an Excel file.")
