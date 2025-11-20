import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Page config ----------
st.set_page_config(
    page_title="Portfolio Auto Analyzer (Excel)",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Styling
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
            table = df_raw.iloc[idx + 1 :].copy()
            table.columns = header
            table = table.dropna(how="all")
            return table, idx

    first_non_empty = df_raw.dropna(how="all").index[0]
    header = df_raw.iloc[first_non_empty]
    table = df_raw.iloc[first_non_empty + 1 :].copy()
    table.columns = header
    table = table.dropna(how="all")
    return table, first_non_empty


def classify_from_scheme_name(name: str) -> str:
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
            "equity", "flexi", "flexicap", "multi cap", "multicap", "mid cap",
            "midcap", "small cap", "smallcap", "large cap", "largecap",
            "index", "elss", "focused", "value fund"
        ]
    ):
        return "Equity"
    return "Other"


def split_main_sub(cat_text: str):
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


def apply_section_subcategories(df_no_total, scheme_col):
    if scheme_col is None:
        return df_no_total, False

    s = df_no_total[scheme_col].astype(str)
    header_mask = s.str.contains(":", regex=False) & ~s.str.contains("TOTAL", case=False)

    if header_mask.sum() == 0:
        return df_no_total, False

    df2 = df_no_total.copy()
    df2["SubCategory"] = s.where(header_mask).ffill()
    df2["MainCategory"] = df2["SubCategory"].apply(lambda x: str(x).split(":", 1)[0].strip())
    df2 = df2[~header_mask].copy()

    return df2, True


def format_inr(x):
    try:
        return f"₹{x:,.0f}"
    except:
        return f"{x}"


# ---------------------- APP UI ----------------------

st.title("📊 Portfolio Auto Analyzer")

uploaded = st.file_uploader("Upload portfolio Excel (SLA / Investwell)", type=["xlsx", "xls"])

if not uploaded:
    st.info("Upload the Valuation / Summary Excel file to begin.")
    st.stop()

try:
    df_full = pd.read_excel(uploaded, sheet_name=0, header=None)
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

if df_full.empty:
    st.error("The first sheet seems empty.")
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
subcat_col = find_col_like(df.columns, ["SUB CATEGORY", "SUBCATEGORY", "TYPE", "ASSET", "CATEGORY"])

# clean numeric
for col in [purchase_col, current_col, gain_col, abs_ret_col, cagr_col]:
    if col is not None:
        try:
            df[col] = num(df[col])
        except:
            pass

# remove TOTAL rows
mask_total = df.apply(lambda r: any("TOTAL" in str(x).upper() for x in r), axis=1)
df_no_total = df[~mask_total].copy()

# derive categories
df_no_total, used_section_style = apply_section_subcategories(df_no_total, scheme_col)

if not used_section_style:
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


# ------------------------------------------------------
# ⭐ NEW: NORMALIZE SUB-CATEGORIES FOR ACCURATE SUGGESTIONS
# ------------------------------------------------------

normalize_map = {
    "smallcap": "Small Cap",
    "small cap": "Small Cap",
    "small-cap": "Small Cap",
    "small cap fund": "Small Cap",

    "midcap": "Mid Cap",
    "mid cap": "Mid Cap",
    "mid-cap": "Mid Cap",

    "largecap": "Large Cap",
    "large cap": "Large Cap",
    "large-cap": "Large Cap",

    "flexicap": "Flexi Cap",
    "flexi cap": "Flexi Cap",
    "flexi-cap": "Flexi Cap",
    "flexi": "Flexi Cap",
}


def normalize_subcat(x):
    s = str(x).lower().strip()
    for key, val in normalize_map.items():
        if key in s:
            return val
    return x  # fallback


df_no_total["SubCategory"] = df_no_total["SubCategory"].apply(normalize_subcat)


# ------------------------------------------------------
# 1. Summary
# ------------------------------------------------------
st.markdown("### 1️⃣ Portfolio Summary")

total_purchase = df_no_total[purchase_col].sum() if purchase_col else 0.0
total_current = df_no_total[current_col].sum() if current_col else 0.0
total_gain = total_current - total_purchase

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


# ------------------------------------------------------
# 2. Category Allocation
# ------------------------------------------------------
st.markdown("### 2️⃣ Category Allocation")

if current_col:
    cat_group_val = df_no_total.groupby("MainCategory")[current_col].sum()
    total_current = df_no_total[current_col].sum()
    cat_group_pct = (cat_group_val / total_current * 100).round(2)

    cat_table = pd.DataFrame({
        "Category": cat_group_val.index,
        "Current Value (₹)": cat_group_val.values,
        "Allocation (%)": cat_group_pct.values,
    }).sort_values("Current Value (₹)", ascending=False)

    col_table, col_chart = st.columns([3, 1])
    with col_table:
        st.dataframe(cat_table, use_container_width=True)

    with col_chart:
        fig, ax = plt.subplots(figsize=(2.4, 2.4))
        ax.pie(cat_group_val.values, autopct="%1.1f%%")
        ax.set_title("Category\nAllocation", fontsize=9)
        st.pyplot(fig)


# ------------------------------------------------------
# 3. Sub-category Allocation
# ------------------------------------------------------
st.markdown("### 3️⃣ Sub-category Allocation")

if current_col:
    sub_group_val = df_no_total.groupby("SubCategory")[current_col].sum()
    total_current = df_no_total[current_col].sum()
    sub_group_pct = (sub_group_val / total_current * 100).round(2)

    sub_table = pd.DataFrame({
        "Sub-Category": sub_group_val.index,
        "Current Value (₹)": sub_group_val.values,
        "Allocation (%)": sub_group_pct.values,
    }).sort_values("Current Value (₹)", ascending=False)

    col_table, col_chart = st.columns([3, 1])
    with col_table:
        st.dataframe(sub_table, use_container_width=True)

    with col_chart:
        fig, ax = plt.subplots(figsize=(2.3, 2.3))
        ax.pie(sub_group_val.values, autopct="%1.1f%%")
        ax.set_title("Sub-category\nAllocation", fontsize=9)
        st.pyplot(fig)


# ------------------------------------------------------
# 4. Scheme Allocation
# ------------------------------------------------------
st.markdown("### 4️⃣ Scheme Allocation (Top 10 by value)")

if current_col and scheme_col:
    alloc = df_no_total[[scheme_col, current_col]].dropna()
    alloc_group = alloc.groupby(scheme_col)[current_col].sum().sort_values(ascending=False)
    total_current = df_no_total[current_col].sum()
    alloc_pct = (alloc_group / total_current * 100).round(2)

    alloc_table = pd.DataFrame({
        "Scheme": alloc_group.index,
        "Current Value (₹)": alloc_group.values,
        "Allocation (%)": alloc_pct.values,
    })

    st.dataframe(alloc_table, use_container_width=True)

# ------------------------------------------------------
# 5a. Allocation Analysis Table
# ------------------------------------------------------
st.markdown("### 5️⃣a Allocation Analysis Table")

ideal_ranges = {
    "Small Cap": (25, 30),
    "Mid Cap": (25, 30),
    "Large Cap": (30, 50),
    "Flexi Cap": (30, 50),
}

sub_group_val = df_no_total.groupby("SubCategory")[current_col].sum()
total_current_value = df_no_total[current_col].sum()
actual_pct_series = (sub_group_val / total_current_value * 100).round(2)

rows = []

for cat, (low, high) in ideal_ranges.items():
    actual_val = float(sub_group_val.get(cat, 0.0))
    actual_pct = float(actual_pct_series.get(cat, 0.0))

    if actual_pct == 0:
        short_pct = low
        short_amt = total_current_value * short_pct / 100
        suggestion = "Increase Allocation"
        rows.append([cat, actual_val, actual_pct, f"{low}% - {high}%", f"Short {short_pct:.2f}%", short_amt, suggestion])
        continue

    if low <= actual_pct <= high:
        rows.append([cat, actual_val, actual_pct, f"{low}% - {high}%", "OK", 0, "No Action"])
    elif actual_pct > high:
        excess_pct = actual_pct - high
        excess_amt = actual_val * excess_pct / 100
        rows.append([cat, actual_val, actual_pct, f"{low}% - {high}%", f"Excess {excess_pct:.2f}%", excess_amt, "Reduce Allocation"])
    else:
        short_pct = low - actual_pct
        short_amt = total_current_value * short_pct / 100
        rows.append([cat, actual_val, actual_pct, f"{low}% - {high}%", f"Short {short_pct:.2f}%", short_amt, "Increase Allocation"])

alloc_table_df = pd.DataFrame(
    rows,
    columns=[
        "Category", "Actual Value (₹)", "Actual %", "Ideal Range",
        "Short / Excess %", "Short / Excess Amt (₹)", "Suggestion"
    ],
)

st.dataframe(alloc_table_df, use_container_width=True)

/*******************************************************
 * SECTION 5 — SCHEME VALIDATION (IDCW / DIRECT / DIVIDEND)
 *******************************************************/

// List of keywords that indicate problems
const invalidTerms = [
  "IDCW",
  "Dividend",
  "Direct",
  "Payout",
  "Pay Out",
  "Reinvestment",
  "Regular "  // Note: includes space to avoid matching "Reg (G)"
];

// Function to check scheme name for invalid patterns
function checkSchemeValidity(schemeName) {
  // If schemeName already contains "Reg (G)" → safe
  if (schemeName.includes("Reg (G)")) return null;

  // Scan for problematic terms
  for (let term of invalidTerms) {
    if (schemeName.toLowerCase().includes(term.toLowerCase())) {
      return term; // return first matched term
    }
  }
  return null;
}

// Build warning messages for all schemes
let schemeWarnings = [];

schemeData.forEach((row, index) => {
  const schemeName = row["Scheme Name"];

  let foundIssue = checkSchemeValidity(schemeName);

  if (foundIssue) {
    schemeWarnings.push(
      `⚠️ <span style="color:red; font-weight:bold">"${schemeName}" contains "${foundIssue}" — Please verify the plan.</span>`
    );
  }
});

// FINAL OUTPUT FOR SUGGESTION BOX
let validationOutput = "";

if (schemeWarnings.length === 0) {
  validationOutput = `
    <div style="padding:10px; background:#e6ffe6; border-left:5px solid #00b300;">
      ✅ <b>All schemes are correctly tagged as Reg (G). No issues found.</b>
    </div>
  `;
} else {
  validationOutput = `
    <div style="padding:10px; background:#ffe6e6; border-left:5px solid red;">
      <b>🚨 Scheme Validation Issues Found:</b><br><br>
      ${schemeWarnings.join("<br>")}
    </div>
  `;
}

// Insert into the Suggestion Box element
document.getElementById("suggestionBox").innerHTML = validationOutput;
