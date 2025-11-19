import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Page config ----------
st.set_page_config(
    page_title="Portfolio Auto Analyzer (Excel)",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Compact but wider container + zoom ----------
st.markdown(
    """
    <style>
    html { zoom: 1.15; }   /* Zoom-in slightly */

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 1080px !important;   /* wider so tables don't look cut */
        margin: auto;
    }

    .stMetric { padding-bottom: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helper functions ----------
def num(series):
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace("", "0")
        .astype(float)
    )

def find_col_like(columns, substrings):
    cols = [str(c).upper().strip() for c in columns]
    for sub in substrings:
        for orig, upper in zip(columns, cols):
            if sub in upper:
                return orig
    return None

def extract_client_name(df_header):
    for r in range(min(10, df_header.shape[0])):
        for c in range(min(5, df_header.shape[1])):
            val = str(df_header.iat[r, c])
            if val.startswith("Client:"):
                text = val.replace("Client:", "").strip()
                return text.split("(")[0].strip()
    return "N/A"

def extract_table(df_raw):
    for idx in range(df_raw.shape[0]):
        first_cell = str(df_raw.iat[idx, 0]).strip().upper()
        if first_cell in ("SCHEME NAME", "SCHEME"):
            header = df_raw.iloc[idx]
            table = df_raw.iloc[idx + 1:].copy()
            table.columns = header
            return table.dropna(how="all"), idx

    first_non_empty = df_raw.dropna(how="all").index[0]
    header = df_raw.iloc[first_non_empty]
    table = df_raw.iloc[first_non_empty + 1:].copy()
    table.columns = header
    return table.dropna(how="all"), first_non_empty

def classify_from_scheme_name(name):
    s = str(name).lower()
    if "liquid" in s or "overnight" in s: return "Liquid"
    if any(k in s for k in ["gilt","debt","bond","corporate"]): return "Debt"
    if "hybrid" in s or "balanced" in s: return "Hybrid"
    if any(k in s for k in ["equity","mid","small","large","flexi","index"]): return "Equity"
    return "Other"

def split_main_sub(cat_text):
    s = str(cat_text).strip()
    if ":" in s:
        main, sub = s.split(":", 1)
        return main.strip(), sub.strip()
    s_low = s.lower()
    if "equity" in s_low: return "Equity", s
    if "debt" in s_low: return "Debt", s
    if "liquid" in s_low: return "Liquid", s
    if "hybrid" in s_low: return "Hybrid", s
    return "Other", s

# ---------- UI ----------
st.title("📊 Portfolio Auto Analyzer")

uploaded = st.file_uploader("Upload portfolio Excel (SLA / Investwell)", type=["xlsx", "xls"])

if not uploaded:
    st.info("Upload Excel file to begin.")
    st.stop()

# Read file
df_full = pd.read_excel(uploaded, sheet_name=0, header=None)

with st.expander("Preview raw sheet (first 30 rows)", expanded=False):
    st.dataframe(df_full.head(30))

client_name = extract_client_name(df_full)
table, header_row = extract_table(df_full)
df = table.copy()

# detect columns
purchase_col = find_col_like(df.columns, ["PURCHASE VALUE", "PURCHASE"])
current_col  = find_col_like(df.columns, ["CURRENT VALUE"])
gain_col     = find_col_like(df.columns, ["GAIN"])
abs_ret_col  = find_col_like(df.columns, ["ABSOLUTE"])
cagr_col     = find_col_like(df.columns, ["CAGR","XIRR"])
scheme_col   = find_col_like(df.columns, ["SCHEME"])
subcat_col   = find_col_like(df.columns, ["CATEGORY","SUB CATEGORY"])

# clean numeric
for col in [purchase_col, current_col, gain_col, abs_ret_col, cagr_col]:
    if col and df[col].dtype == "object":
        df[col] = num(df[col])

# remove TOTAL row
mask_total = df.apply(lambda r: any("TOTAL" in str(x).upper() for x in r), axis=1)
df_no_total = df[~mask_total].copy()

# derive categories
if subcat_col:
    main_sub = df_no_total[subcat_col].apply(split_main_sub)
    df_no_total["MainCategory"] = main_sub.apply(lambda x: x[0])
    df_no_total["SubCategory"] = main_sub.apply(lambda x: x[1])
else:
    df_no_total["MainCategory"] = df_no_total[scheme_col].apply(classify_from_scheme_name)
    df_no_total["SubCategory"] = df_no_total["MainCategory"]

# ---------- 1. Summary ----------
st.markdown("### 1️⃣ Portfolio Summary")

total_purchase = df_no_total[purchase_col].sum() if purchase_col else 0
total_current  = df_no_total[current_col].sum() if current_col else 0
total_gain     = total_current - total_purchase

c1, c2, c3 = st.columns(3)
c1.metric("Client", client_name)
c2.metric("Purchase (₹)", f"{total_purchase:,.0f}")
c3.metric("Current (₹)", f"{total_current:,.0f}")

c4, c5 = st.columns(2)
c4.metric("Gain / Loss (₹)", f"{total_gain:,.0f}")
if abs_ret_col:
    c5.metric("Avg. Abs Return (%)", f"{df_no_total[abs_ret_col].mean():.2f}")
elif cagr_col:
    c5.metric("Avg. CAGR (%)", f"{df_no_total[cagr_col].mean():.2f}")
else:
    c5.metric("Avg Return", "N/A")

with st.expander("Scheme-level table", expanded=False):
    st.dataframe(df_no_total.reset_index(drop=True))

# ---------- 2. Category Allocation ----------
st.markdown("### 2️⃣ Category Allocation")

if current_col:
    cat_val = df_no_total.groupby("MainCategory")[current_col].sum()
    cat_pct = (cat_val / total_current * 100).round(2)

    table = pd.DataFrame({
        "Category": cat_val.index,
        "Current Value (₹)": cat_val.values,
        "Allocation (%)": cat_pct.values
    })

    col1, col2 = st.columns([3,1])
    col1.dataframe(table, use_container_width=True)

    fig, ax = plt.subplots(figsize=(2.4,2.4))
    ax.pie(cat_val.values, autopct="%1.1f%%")
    ax.set_title("Category\nAllocation", fontsize=9)
    col2.pyplot(fig)

# ---------- 3. Sub-category Allocation ----------
st.markdown("### 3️⃣ Sub-category Allocation")

if current_col:
    sub_val = df_no_total.groupby("SubCategory")[current_col].sum()
    sub_pct = (sub_val / total_current * 100).round(2)

    table = pd.DataFrame({
        "Sub-Category": sub_val.index,
        "Current Value (₹)": sub_val.values,
        "Allocation (%)": sub_pct.values
    })

    col1, col2 = st.columns([3,1])
    col1.dataframe(table, use_container_width=True)

    fig, ax = plt.subplots(figsize=(2.3,2.3))
    ax.pie(sub_val.values, autopct="%1.1f%%")
    ax.set_title("Sub-category\nAllocation", fontsize=9)
    col2.pyplot(fig)

# ---------- 4. Scheme Allocation (TABLE ONLY) ----------
st.markdown("### 4️⃣ Scheme Allocation (Table Only)")

if current_col and scheme_col:
    alloc = df_no_total.groupby(scheme_col)[current_col].sum().sort_values(ascending=False)
    pct   = (alloc / total_current * 100).round(2)

    final = pd.DataFrame({
        "Scheme": alloc.index,
        "Current Value (₹)": alloc.values,
        "Allocation (%)": pct.values,
    })

    st.dataframe(final, use_container_width=True)
else:
    st.info("Could not detect Scheme / Current Value columns.")
