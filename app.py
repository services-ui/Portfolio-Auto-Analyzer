import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Page config (CENTERED, SIMPLE VIEW) ----------
st.set_page_config(
    page_title="Portfolio Auto Analyzer (Excel)",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# global zoom (zoom in page a bit more)
st.markdown(
    """
    <style>
    html {
        zoom: 1.20;  /* 20% zoom-in; adjust to 1.15 or 1.25 if needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# compact centered container
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 960px;      /* centered, not full screen */
        margin: auto;
    }
    .stMetric {
        padding-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    """Find first column whose name contains any of the given substrings (case-insensitive)."""
    cols = [str(c).upper().strip() for c in columns]
    for sub in substrings:
        for orig, upper in zip(columns, cols):
            if sub in upper:
                return orig
    return None


def extract_client_name(df_header):
    """Look for cell starting with 'Client:' in first few rows/cols and extract the name part."""
    for r in range(min(10, df_header.shape[0])):
        for c in range(min(5, df_header.shape[1])):
            val = str(df_header.iat[r, c])
            if val.startswith("Client:"):
                text = val.replace("Client:", "").strip()
                if "(" in text:  # remove PAN in brackets if present
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
            table = table.dropna(how="all")  # drop fully empty rows
            return table, idx
    # Fallback: first non-empty row
    first_non_empty = df_raw.dropna(how="all").index[0]
    header = df_raw.iloc[first_non_empty]
    table = df_raw.iloc[first_non_empty + 1 :].copy()
    table.columns = header
    table = table.dropna(how="all")
    return table, first_non_empty


def classify_from_scheme_name(name: str) -> str:
    """Fallback classification if we don't find a category column."""
    s = str(name).lower()

    if "liquid" in s or "overnight" in s or "money market" in s:
        return "Liquid"
    if any(k in s for k in ["gilt", "debt", "bond", "income", "credit risk", "corporate bond"]):
        return "Debt"
    if "hybrid" in s or "balanced" in s or "aggressive hybrid" in s:
        return "Hybrid"
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


def split_main_sub(cat_text: str):
    """From 'Equity: Small Cap' return ('Equity', 'Small Cap')."""
    s = str(cat_text).strip()
    if ":" in s:
        main, sub = s.split(":", 1)
        return main.strip(), sub.strip()
    lower = s.lower()
    if "equity" in lower:
        return "Equity", s
    if "debt" in lower or "bond" in lower or "gilt" in lower:
        return "Debt", s
    if "liquid" in lower or "overnight" in lower or "money market" in lower:
        return "Liquid", s
    if "hybrid" in lower or "balanced" in lower:
        return "Hybrid", s
    return "Other", s


# ---------- App UI ----------

st.title("📊 Portfolio Auto Analyzer")

uploaded = st.file_uploader("Upload portfolio Excel (SLA / Investwell)", type=["xlsx", "xls"])

if not uploaded:
    st.info("Upload the Valuation / Summary Excel file to begin.")
    st.stop()

# Read first sheet without header
try:
    df_full = pd.read_excel(uploaded, sheet_name=0, header=None)
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

if df_full.empty:
    st.error("The first sheet seems to be empty.")
    st.stop()

with st.expander("Preview raw sheet (first 30 rows)", expanded=False):
    st.dataframe(df_full.head(30))

client_name = extract_client_name(df_full)
table, header_row = extract_table(df_full)
df = table.copy()

# detect columns
purchase_col = find_col_like(df.columns, ["PURCHASE VALUE", "PURCHASE OUTSTANDING", "PURCHASE"])
current_col = find_col_like(df.columns, ["CURRENT VALUE"])
gain_col = find_col_like(df.columns, ["GAIN", "REALIZED GAIN"])
abs_ret_col = find_col_like(df.columns, ["ABSOLUTE"])
cagr_col = find_col_like(df.columns, ["CAGR", "XIRR"])
scheme_col = find_col_like(df.columns, ["SCHEME", "SCHEME NAME"])
subcat_col = find_col_like(df.columns, ["CATEGORY", "SUB CATEGORY", "SUBCATEGORY", "TYPE", "ASSET"])

# clean numeric
for col in [purchase_col, current_col, gain_col, abs_ret_col, cagr_col]:
    if col is not None and df[col].dtype not in ("float64", "int64"):
        try:
            df[col] = num(df[col])
        except Exception:
            pass

# remove GRAND TOTAL row
mask_total = df.apply(lambda r: any("TOTAL" in str(x).upper() for x in r), axis=1)
df_no_total = df[~mask_total].copy()

# derive categories
if subcat_col:
    main_sub = df_no_total[subcat_col].apply(split_main_sub)
    df_no_total["MainCategory"] = main_sub.apply(lambda x: x[0])
    df_no_total["SubCategory"] = main_sub.apply(lambda x: x[1])
elif scheme_col:
    df_no_total["MainCategory"] = df_no_total[scheme_col].apply(classify_from_scheme_name)
    df_no_total["SubCategory"] = df_no_total["MainCategory"]
else:
    df_no_total["MainCategory"] = "Other"
    df_no_total["SubCategory"] = "Other"

# ---------- 1. Summary ----------

st.markdown("### 1️⃣ Portfolio Summary")

total_purchase = df_no_total[purchase_col].sum() if purchase_col else 0.0
total_current = df_no_total[current_col].sum() if current_col else 0.0
total_gain = total_current - total_purchase if purchase_col and current_col else 0.0

avg_abs = df_no_total[abs_ret_col].mean() if abs_ret_col else None
avg_cagr = df_no_total[cagr_col].mean() if cagr_col else None

c1, c2, c3 = st.columns(3)
c1.metric("Client", client_name)
c2.metric("Purchase (₹)", f"{total_purchase:,.0f}")
c3.metric("Current (₹)", f"{total_current:,.0f}")

c4, c5 = st.columns(2)
c4.metric("Gain / Loss (₹)", f"{total_gain:,.0f}")
if avg_abs is not None:
    c5.metric("Avg. Abs Return (%)", f"{avg_abs:.2f}")
elif avg_cagr is not None:
    c5.metric("Avg. CAGR / XIRR (%)", f"{avg_cagr:.2f}")
else:
    c5.metric("Avg. Return", "N/A")

with st.expander("Scheme-level table", expanded=False):
    st.dataframe(df_no_total.reset_index(drop=True))

# ---------- 2. Category-wise Allocation ----------

st.markdown("### 2️⃣ Category Allocation")

if current_col:
    cat_group_val = df_no_total.groupby("MainCategory")[current_col].sum()
    cat_group_pct = (cat_group_val / total_current * 100).round(2) if total_current > 0 else 0

    cat_table = pd.DataFrame(
        {
            "Category": cat_group_val.index.astype(str),
            "Current Value (₹)": cat_group_val.values,
            "Allocation (%)": cat_group_pct.values,
        }
    ).sort_values("Current Value (₹)", ascending=False)

    col_table, col_chart = st.columns([3, 1])
    with col_table:
        st.dataframe(cat_table, use_container_width=True)

    with col_chart:
        fig_cat, ax_cat = plt.subplots(figsize=(2.4, 2.4))
        ax_cat.pie(cat_group_val.values, labels=None, autopct="%1.1f%%")
        ax_cat.set_title("Category\nAllocation", fontsize=9)
        st.pyplot(fig_cat)
else:
    st.info("Could not detect Current Value column for category allocation.")

# ---------- 3. Sub-category Allocation ----------

st.markdown("### 3️⃣ Sub-category Allocation")

if current_col:
    sub_group_val = df_no_total.groupby("SubCategory")[current_col].sum()
    sub_group_pct = (sub_group_val / total_current * 100).round(2) if total_current > 0 else 0

    sub_table = pd.DataFrame(
        {
            "Sub-Category": sub_group_val.index.astype(str),
            "Current Value (₹)": sub_group_val.values,
            "Allocation (%)": sub_group_pct.values,
        }
    ).sort_values("Current Value (₹)", ascending=False)

    col_table, col_chart = st.columns([3, 1])
    with col_table:
        st.dataframe(sub_table, use_container_width=True)

    with col_chart:
        fig_sub, ax_sub = plt.subplots(figsize=(2.3, 2.3))
        ax_sub.pie(sub_group_val.values, labels=None, autopct="%1.1f%%")
        ax_sub.set_title("Sub-category\nAllocation", fontsize=9)
        st.pyplot(fig_sub)
else:
    st.info("Could not detect Current Value column for sub-category allocation.")

# ---------- 4. Scheme-wise Allocation (TABLE ONLY) ----------

st.markdown("### 4️⃣ Scheme Allocation (Table Only)")

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

    st.dataframe(alloc_table, use_container_width=True)

else:
    st.info("Could not detect Scheme / Current Value columns for scheme allocation.")
