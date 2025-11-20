import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Page config ----------
st.set_page_config(
    page_title="Portfolio Auto Analyzer (Excel)",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# UI compact layout
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 960px;
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
                if "(" in text:
                    name = text.split("(")[0].strip()
                else:
                    name = text
                return name
    return "N/A"

def extract_table(df_raw):
    for idx in range(df_raw.shape[0]):
        first_cell = str(df_raw.iat[idx, 0]).strip().upper()
        if first_cell in ("SCHEME NAME", "SCHEME"):
            header = df_raw.iloc[idx]
            table = df_raw.iloc[idx+1:].copy()
            table.columns = header
            table = table.dropna(how="all")
            return table, idx

    first_non_empty = df_raw.dropna(how="all").index[0]
    header = df_raw.iloc[first_non_empty]
    table = df_raw.iloc[first_non_empty + 1:].copy()
    table.columns = header
    return table, first_non_empty

def classify_from_scheme_name(name: str) -> str:
    s = str(name).lower()

    if "liquid" in s or "overnight" in s or "money market" in s:
        return "Liquid"
    if any(k in s for k in ["gilt", "debt", "bond", "income", "credit"]):
        return "Debt"
    if "hybrid" in s or "balanced" in s:
        return "Hybrid"
    if any(k in s for k in [
        "equity", "flexi", "multi", "mid", "small", "large", "index", "elss", "focused"
    ]):
        return "Equity"
    return "Other"

def split_main_sub(cat_text: str):
    s = str(cat_text).strip()
    if ":" in s:
        main, sub = s.split(":", 1)
        return main.strip(), sub.strip()

    lower = s.lower()
    if "equity" in lower: return "Equity", s
    if "debt" in lower: return "Debt", s
    if "liquid" in lower: return "Liquid", s
    if "hybrid" in lower: return "Hybrid", s
    return "Other", s

def apply_section_subcategories(df_no_total, scheme_col):
    s = df_no_total[scheme_col].astype(str)
    header_mask = s.str.contains(":", regex=False) & ~s.str.contains("TOTAL", case=False)

    if header_mask.sum() == 0:
        return df_no_total, False

    df2 = df_no_total.copy()
    df2["SubCategory"] = s.where(header_mask).ffill()
    df2["MainCategory"] = df2["SubCategory"].apply(lambda x: str(x).split(":", 1)[0].strip())
    df2 = df2[~header_mask].copy()

    return df2, True

# ---------- App Main UI ----------
st.title("📊 Portfolio Auto Analyzer")

uploaded = st.file_uploader("Upload portfolio Excel", type=["xlsx", "xls"])

if not uploaded:
    st.stop()

# Read Excel
df_full = pd.read_excel(uploaded, sheet_name=0, header=None)
client_name = extract_client_name(df_full)
table, header_row = extract_table(df_full)
df = table.copy()

# Find columns
purchase_col = find_col_like(df.columns, ["PURCHASE"])
current_col  = find_col_like(df.columns, ["CURRENT VALUE"])
abs_ret_col  = find_col_like(df.columns, ["ABSOLUTE"])
cagr_col     = find_col_like(df.columns, ["CAGR", "XIRR"])
scheme_col   = find_col_like(df.columns, ["SCHEME"])
subcat_col   = find_col_like(df.columns, ["CATEGORY", "SUB CATEGORY"])
amc_col      = find_col_like(df.columns, ["AMC", "FUND HOUSE", "FUND"])

# Clean numeric
for col in [purchase_col, current_col, abs_ret_col, cagr_col]:
    if col:
        df[col] = num(df[col])

# Remove "GRAND TOTAL"
mask_total = df.apply(lambda r: "TOTAL" in str(r).upper(), axis=1)
df_no_total = df[~mask_total].copy()

# Category & Sub-category
df_no_total, used_section_style = apply_section_subcategories(df_no_total, scheme_col)

# ---------- 1. Summary ----------
st.markdown("### 1️⃣ Portfolio Summary")

total_purchase = df_no_total[purchase_col].sum()
total_current = df_no_total[current_col].sum()
total_gain = total_current - total_purchase

c1, c2, c3 = st.columns(3)
c1.metric("Client", client_name)
c2.metric("Purchase (₹)", f"{total_purchase:,.0f}")
c3.metric("Current (₹)", f"{total_current:,.0f}")

# ---------- 2. Category Allocation ----------
st.markdown("### 2️⃣ Category Allocation")

cat_group_val = df_no_total.groupby("MainCategory")[current_col].sum()
cat_group_pct = cat_group_val / total_current * 100

col_table, col_chart = st.columns([3, 1])
col_table.dataframe(
    pd.DataFrame({
        "Category": cat_group_val.index,
        "Current Value (₹)": cat_group_val.values,
        "Allocation (%)": cat_group_pct.round(2).values,
    })
)

fig1, ax1 = plt.subplots(figsize=(2.3, 2.3))
ax1.pie(cat_group_val.values, autopct="%1.1f%%")
col_chart.pyplot(fig1)

# ---------- 3. Sub-category Allocation ----------
st.markdown("### 3️⃣ Sub-category Allocation")

sub_group_val = df_no_total.groupby("SubCategory")[current_col].sum()
sub_group_pct = sub_group_val / total_current * 100

col_table, col_chart = st.columns([3, 1])
col_table.dataframe(
    pd.DataFrame({
        "Sub-Category": sub_group_val.index,
        "Current Value (₹)": sub_group_val.values,
        "Allocation (%)": sub_group_pct.round(2).values,
    })
)

fig2, ax2 = plt.subplots(figsize=(2.3, 2.3))
ax2.pie(sub_group_val.values, autopct="%1.1f%%")
col_chart.pyplot(fig2)

# ---------- 4. Scheme Allocation ----------
st.markdown("### 4️⃣ Scheme Allocation (Top 10)")

alloc = df_no_total.groupby(scheme_col)[current_col].sum().sort_values(ascending=False)
st.dataframe(
    pd.DataFrame({
        "Scheme": alloc.index,
        "Current Value (₹)": alloc.values,
        "Allocation (%)": (alloc / total_current * 100).round(2).values,
    })
)

# ---------- 5. Suggestions ----------
st.markdown("### 5️⃣ Smart Suggestions 🔍")

suggestions = []

# ---- A. Sub-category Ideal Allocation Check ----
ideal_ranges = {
    "Small Cap": (25, 30),
    "Mid Cap": (25, 30),
    "Large Cap": (30, 50),
    "Flexi Cap": (30, 50),
}

for subcat, pct in sub_group_pct.items():
    if subcat in ideal_ranges:
        low, high = ideal_ranges[subcat]

        if pct < low:
            suggestions.append(f"🔸 **{subcat} is UNDER-allocated** ({pct:.2f}%). Ideal: {low}-{high}%")
        elif pct > high:
            suggestions.append(f"🔴 **{subcat} is OVER-allocated** ({pct:.2f}%). Ideal: {low}-{high}%")
        else:
            suggestions.append(f"🟢 {subcat} is within ideal range ({pct:.2f}%).")

# ---- B. AMC Exposure > 20% ----
if amc_col:
    amc_group = df_no_total.groupby(amc_col)[current_col].sum()
    amc_pct = amc_group / total_current * 100

    for amc, pct in amc_pct.items():
        if pct > 20:
            suggestions.append(f"🔴 **High AMC Exposure:** {amc} = {pct:.2f}% (>20%)")

# ---- C. Non 'Regular Growth' Schemes ----
bad_schemes = []
for s in df_no_total[scheme_col]:
    name = str(s).lower()
    if not ("regular" in name and "growth" in name):
        bad_schemes.append(s)

if bad_schemes:
    suggestions.append(
        f"🔴 **Non Regular-Growth Schemes Found:** {', '.join(bad_schemes[:10])}..."
    )

# Display suggestions
if suggestions:
    for s in suggestions:
        st.markdown(f"- {s}")
else:
    st.success("All allocations are healthy ✔")
