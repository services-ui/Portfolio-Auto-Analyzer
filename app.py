import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Auto Analyzer (Excel)", layout="wide")

# ---------- Helper functions ----------

def num(series):
    """Convert a column to numeric if possible."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace("", "0")
        .astype(float)
    )


def find_col_like(columns, substrings):
    """
    Find first column whose name contains any of the given substrings (case-insensitive).
    substrings: list of UPPERCASE fragments.
    """
    cols = [str(c).upper().strip() for c in columns]
    for sub in substrings:
        for orig, upper in zip(columns, cols):
            if sub in upper:
                return orig
    return None


def extract_client_name(df_header):
    """
    Look for cell starting with 'Client:' in first few rows/cols and extract the name part.
    """
    for r in range(min(10, df_header.shape[0])):
        for c in range(min(5, df_header.shape[1])):
            val = str(df_header.iat[r, c])
            if val.startswith("Client:"):
                text = val.replace("Client:", "").strip()
                # Remove PAN in brackets if present
                if "(" in text:
                    name = text.split("(")[0].strip()
                else:
                    name = text
                return name
    return "N/A"


def extract_table(df_raw):
    """
    From a headerless sheet (header=None), find the row where first cell is
    'SCHEME NAME' or 'SCHEME', treat that as header, and return (table_df, header_row_index).
    """
    for idx in range(df_raw.shape[0]):
        first_cell = str(df_raw.iat[idx, 0]).strip().upper()
        if first_cell in ("SCHEME NAME", "SCHEME"):
            header = df_raw.iloc[idx]
            table = df_raw.iloc[idx + 1 :].copy()
            table.columns = header
            # Drop fully empty rows
            table = table.dropna(how="all")
            return table, idx
    # Fallback: assume first non-empty row is header
    first_non_empty = df_raw.dropna(how="all").index[0]
    header = df_raw.iloc[first_non_empty]
    table = df_raw.iloc[first_non_empty + 1 :].copy()
    table.columns = header
    table = table.dropna(how="all")
    return table, first_non_empty


def classify_from_scheme_name(name: str) -> str:
    """
    Classify category (Equity / Debt / Liquid / Hybrid / Other)
    based ONLY on text in scheme name column.
    """
    s = str(name).lower()

    # Liquid / money market / overnight
    if "liquid" in s or "overnight" in s or "money market" in s:
        return "Liquid"

    # Debt-style keywords
    if any(k in s for k in ["gilt", "debt", "bond", "income", "credit risk", "corporate bond"]):
        return "Debt"

    # Hybrid / balanced
    if "hybrid" in s or "balanced" in s or "aggressive hybrid" in s:
        return "Hybrid"

    # Equity style (flexi / large / mid / small / index / elss / focused etc.)
    if any(
        k in s
        for k in [
            "equity",
            "flexi",
            "flexicap",
            "multi cap",
            "multicap",
            "mid cap",
            "midcap",
            "small cap",
            "smallcap",
            "large cap",
            "largecap",
            "index",
            "elss",
            "focused",
            "value fund",
        ]
    ):
        return "Equity"

    return "Other"


# ---------- UI ----------

st.title("📊 Portfolio Auto Analyzer (Excel – Scheme Level)")
st.write("Upload your **Valuation / Summary Excel** from SLA / Investwell (the ones you showed).")

uploaded = st.file_uploader("Upload Excel", type=["xlsx", "xls"])

if not uploaded:
    st.info("Please upload the Excel file.")
    st.stop()

# Read entire first sheet with NO header so we can find header row & client name
try:
    df_full = pd.read_excel(uploaded, sheet_name=0, header=None)
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

if df_full.empty:
    st.error("The first sheet seems to be empty.")
    st.stop()

# Preview raw data
with st.expander("Preview raw sheet (first 40 rows)"):
    st.dataframe(df_full.head(40))

# Extract client name from the "Client: ..." line
client_name = extract_client_name(df_full)

# Extract the main table starting at SCHEME NAME / SCHEME
table, header_row = extract_table(df_full)

# Keep a working copy
df = table.copy()

# Detect key columns based on your formats
purchase_col = find_col_like(df.columns, ["PURCHASE VALUE", "PURCHASE OUTSTANDING", "PURCHASE"])
current_col = find_col_like(df.columns, ["CURRENT VALUE"])
gain_col = find_col_like(df.columns, ["GAIN", "REALIZED GAIN"])
abs_ret_col = find_col_like(df.columns, ["ABSOLUTE"])
cagr_col = find_col_like(df.columns, ["CAGR", "XIRR"])
scheme_col = find_col_like(df.columns, ["SCHEME", "SCHEME NAME"])

# Clean numeric columns if present
for col in [purchase_col, current_col, gain_col, abs_ret_col, cagr_col]:
    if col is not None:
        try:
            df[col] = num(df[col])
        except Exception:
            pass

# Remove possible GRAND TOTAL row (contains 'TOTAL' text somewhere)
mask_total = df.apply(lambda r: any("TOTAL" in str(x).upper() for x in r), axis=1)
df_no_total = df[~mask_total].copy()


# ---------- 1. Portfolio Summary ----------

st.markdown("---")
st.header("1️⃣ Portfolio Summary")

total_purchase = df_no_total[purchase_col].sum() if purchase_col else 0.0
total_current = df_no_total[current_col].sum() if current_col else 0.0
total_gain = total_current - total_purchase if purchase_col and current_col else 0.0

avg_abs = df_no_total[abs_ret_col].mean() if abs_ret_col else None
avg_cagr = df_no_total[cagr_col].mean() if cagr_col else None

c1, c2, c3 = st.columns(3)
c1.metric("Client Name", client_name)
c2.metric("Total Purchase Value (₹)", f"{total_purchase:,.0f}")
c3.metric("Total Current Value (₹)", f"{total_current:,.0f}")

c4, c5 = st.columns(2)
c4.metric("Total Gain / Loss (₹)", f"{total_gain:,.0f}")
if avg_abs is not None:
    c5.metric("Average Absolute Return (%)", f"{avg_abs:.2f}")
elif avg_cagr is not None:
    c5.metric("Average CAGR / XIRR (%)", f"{avg_cagr:.2f}")
else:
    c5.metric("Average Return", "N/A")

st.write("Underlying scheme-level table (without GRAND TOTAL row):")
st.dataframe(df_no_total.reset_index(drop=True))


# ---------- 2. Category-wise amount from Scheme Name ----------

st.markdown("---")
st.header("2️⃣ Category-wise Amount (from Scheme Name)")

if scheme_col and current_col:
    cat_df = df_no_total[[scheme_col, current_col]].dropna(subset=[scheme_col])
    cat_df["Category"] = cat_df[scheme_col].apply(classify_from_scheme_name)

    cat_group_val = cat_df.groupby("Category")[current_col].sum()
    if total_current > 0:
        cat_group_pct = (cat_group_val / total_current * 100).round(2)
    else:
        cat_group_pct = cat_group_val * 0

    cat_table = pd.DataFrame(
        {
            "Category": cat_group_val.index.astype(str),
            "Current Value (₹)": cat_group_val.values,
            "Allocation (%)": cat_group_pct.values,
        }
    ).sort_values("Current Value (₹)", ascending=False)

    st.dataframe(cat_table)

    fig_cat, ax_cat = plt.subplots()
    ax_cat.bar(cat_table["Category"], cat_table["Current Value (₹)"])
    ax_cat.set_xticklabels(cat_table["Category"], rotation=45, ha="right")
    ax_cat.set_ylabel("Current Value (₹)")
    ax_cat.set_title("Category-wise Amount (from Scheme Name)")
    st.pyplot(fig_cat)

else:
    st.info("Could not detect Scheme / Current Value columns to build category-wise allocation.")


# ---------- 3. Scheme-wise Allocation ----------

st.markdown("---")
st.header("3️⃣ Scheme-wise Allocation (by Current Value)")

if current_col and scheme_col:
    alloc = df_no_total[[scheme_col, current_col]].dropna()
    alloc_group = alloc.groupby(scheme_col)[current_col].sum().sort_values(ascending=False)

    alloc_pct = (alloc_group / total_current * 100).round(2) if total_current > 0 else alloc_group * 0

    alloc_table = pd.DataFrame(
        {
            "Scheme": alloc_group.index.astype(str),
            "Current Value (₹)": alloc_group.values,
            "Allocation (%)": alloc_pct.values,
        }
    )

    st.dataframe(alloc_table)

    # Bar chart of top 10 schemes
    top = alloc_table.head(10)
    fig, ax = plt.subplots()
    ax.bar(top["Scheme"], top["Allocation (%)"])
    ax.set_xticklabels(top["Scheme"], rotation=45, ha="right")
    ax.set_ylabel("Allocation (%)")
    ax.set_title("Top Schemes by Allocation (Current Value)")
    st.pyplot(fig)
else:
    st.info("Could not detect Scheme / Current Value columns to build allocation chart.")
