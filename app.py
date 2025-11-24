
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

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
            "equity","flexi","flexicap","multi cap","multicap",
            "mid cap","midcap","small cap","smallcap","large cap","largecap",
            "index","elss","focused","value fund"
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

# ------------------------------------------------------
# INDIAN NUMBERING FORMATTER (Lakhs / Crores) — Always decimals
# ------------------------------------------------------
def inr_format(x):
    try:
        x = float(x)
        s = f"{x:,.2f}"
        parts = s.split(".")
        int_part = parts[0]
        dec_part = parts[1]

        if len(int_part) > 3:
            main = int_part[:-3].replace(",", "")
            last3 = int_part[-3:]
            grouped = ""

            while len(main) > 2:
                grouped = "," + main[-2:] + grouped
                main = main[:-2]

            grouped = main + grouped
            s_int = grouped + "," + last3
        else:
            s_int = int_part

        return f"₹ {s_int}.{dec_part}"

    except:
        return x

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

# clean numeric columns
for col in [purchase_col, current_col, gain_col, abs_ret_col, cagr_col]:
    if col is not None:
        try:
            df[col] = num(df[col])
        except:
            pass

# remove "TOTAL" rows AND client header rows (NAME + PAN)
def is_client_name_pattern(text):
    text = str(text).strip()
    pattern = r"\([A-Z]{5}[0-9]{4}[A-Z]\)"  # (ABCDE1234F)
    return bool(re.search(pattern, text))

mask_total = df.apply(
    lambda r: (
        any("TOTAL" in str(x).upper() for x in r) or
        is_client_name_pattern(r.iloc[0])
    ),
    axis=1
)

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

# normalization map
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
    return x

df_no_total["SubCategory"] = df_no_total["SubCategory"].apply(normalize_subcat)

# ------------------------------------------------------
# 1. Summary (Corrected CAGR from Grand Total)
# ------------------------------------------------------
st.markdown("### 1️⃣ Portfolio Summary")

grand_total_cagr = None

if cagr_col:
    try:
        mask = df.iloc[:, 0].astype(str).str.contains("Grand Total", case=False, na=False)
        if mask.any():
            row = df[mask].iloc[0]
            raw_value = str(row[cagr_col]).replace("%", "").replace(",", "").strip()
            grand_total_cagr = float(raw_value)
    except:
        grand_total_cagr = None

total_purchase = df_no_total[purchase_col].sum() if purchase_col else 0.0
total_current = df_no_total[current_col].sum() if current_col else 0.0
total_gain = total_current - total_purchase

c1, c2, c3 = st.columns(3)
c1.metric("Client", client_name)
c2.metric("Purchase (₹)", inr_format(total_purchase))
c3.metric("Current (₹)", inr_format(total_current))

c4, c5 = st.columns(2)
c4.metric("Gain / Loss (₹)", inr_format(total_gain))

if grand_total_cagr is not None:
    c5.metric("CAGR / XIRR (%)", f"{grand_total_cagr:.2f}%")
else:
    c5.metric("CAGR / XIRR (%)", "N/A")

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

    cat_table["Current Value (₹)"] = cat_table["Current Value (₹)"].apply(inr_format)

    col_table, col_chart = st.columns([3,1])

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

    sub_table["Current Value (₹)"] = sub_table["Current Value (₹)"].apply(inr_format)

    col_table, col_chart = st.columns([3,1])

    with col_table:
        st.dataframe(sub_table, use_container_width=True)

    with col_chart:
        fig, ax = plt.subplots(figsize=(2.3, 2.3))
        ax.pie(sub_group_val.values, autopct="%1.1f%%")
        ax.set_title("Sub-category\nAllocation", fontsize=9)
        st.pyplot(fig)

# ------------------------------------------------------
# 4. Scheme Allocation (All Schemes + Risk Score)
# ------------------------------------------------------
st.markdown("### 4️⃣ Scheme Allocation (All Schemes)")

if current_col and scheme_col:
    # Extract scheme name + current value
    alloc = df_no_total[[scheme_col, current_col]].dropna()

    # Group duplicate schemes by name
    alloc_group = (
        alloc.groupby(scheme_col)[current_col]
        .sum()
        .sort_values(ascending=False)
    )

    total_current = df_no_total[current_col].sum()

    # Calculate allocation %
    alloc_pct = (alloc_group / total_current * 100).round(2)

    # Build table rows
    rows = []
    for scheme_name, value in alloc_group.items():
        pct = alloc_pct[scheme_name]

        # Risk logic
        if pct > 10:
            risk = "🟥 High Risk"
        else:
            risk = "🟩 Safe"

        rows.append([scheme_name, value, pct, risk])

    # Final table
    alloc_table = pd.DataFrame(
        rows,
        columns=[
            "Scheme",
            "Total Current Value (₹)",
            "Allocation (%)",
            "Risk Score"
        ]
    )

    # Apply Indian format to value column
    alloc_table["Total Current Value (₹)"] = alloc_table["Total Current Value (₹)"].apply(inr_format)

    st.dataframe(alloc_table, use_container_width=True)


# ------------------------------------------------------
# 4a. AMC-wise Allocation (Table + Expanders)
# ------------------------------------------------------
st.markdown("### 4️⃣a AMC-wise Allocation")

if scheme_col:

    # Extract AMC name from scheme (first word before first space)
    def get_amc_name(s):
        return str(s).split()[0].strip()

    df_no_total["AMC"] = df_no_total[scheme_col].astype(str).apply(get_amc_name)

    amc_group_val = df_no_total.groupby("AMC")[current_col].sum()
    total_current_val = df_no_total[current_col].sum()

    amc_pct = (amc_group_val / total_current_val * 100).round(2)

    # Prepare rows for table
    amc_rows = []
    amc_scheme_map = {}

    for amc in amc_group_val.index:

        value = amc_group_val[amc]
        pct = amc_pct[amc]

        # Risk Scoring
        if pct < 20:
            risk = "🟩 Low Risk"
        elif 20 <= pct <= 25:
            risk = "🟧 Medium Risk"
        else:
            risk = "🟥 High Risk"

        # Schemes under this AMC
        schemes_df = df_no_total[df_no_total["AMC"] == amc]
        scheme_count = schemes_df.shape[0]

        # Store scheme list (for expander)
        amc_scheme_map[amc] = schemes_df[[scheme_col, current_col]]

        amc_rows.append([amc, value, pct, risk, scheme_count])

    # Build AMC Table
    amc_table_df = pd.DataFrame(
        amc_rows,
        columns=[
            "AMC", "Total Value (₹)", "Allocation %", "Risk Score", "Schemes Count"
        ]
    ).sort_values("Total Value (₹)", ascending=False)

    # Format Total Value
    amc_table_df["Total Value (₹)"] = amc_table_df["Total Value (₹)"].apply(inr_format)

    st.dataframe(amc_table_df, use_container_width=True)

    # Expanders for each AMC
    st.markdown("### 🔽 AMC Schemes Breakdown")

    for amc in amc_table_df["AMC"]:
        count_val = amc_table_df.loc[amc_table_df["AMC"] == amc, "Schemes Count"].values[0]
        with st.expander(f"{amc} — Schemes ({count_val})"):
            amc_df = amc_scheme_map[amc].copy()
            amc_df_display = amc_df.rename(
                columns={scheme_col: "Scheme", current_col: "Current Value (₹)"}
            )
            # Format scheme values
            amc_df_display["Current Value (₹)"] = amc_df_display["Current Value (₹)"].apply(inr_format)
            st.dataframe(amc_df_display, use_container_width=True)

else:
    st.info("AMC breakdown unavailable — Scheme column not detected.")


# ------------------------------------------------------
# 5a. Allocation Analysis Table  (UPDATED + Formatted)
# ------------------------------------------------------
st.markdown("### 5️⃣a Allocation Analysis Table")

# Ideal ranges for allocation
ideal_ranges = {
    "Large Cap": (30, 50),
    "Mid Cap": (25, 30),
    "Small Cap": (25, 30),
    "Flexi Cap": (30, 50),

    # NEW RULES (Max-only rules)
    "Value Fund": (0, 10),
    "Focused Fund": (0, 10),
}

# Calculate actual % by sub-category
sub_group_val = df_no_total.groupby("SubCategory")[current_col].sum()
total_current_value = df_no_total[current_col].sum()
actual_pct_series = (sub_group_val / total_current_value * 100).round(2)

rows = []

for cat, (low, high) in ideal_ranges.items():

    actual_val = float(sub_group_val.get(cat, 0.0))
    actual_pct = float(actual_pct_series.get(cat, 0.0))

    # ---- CASE 1: Category not invested at all ----
    if actual_pct == 0 and low > 0:
        short_pct = low
        short_amt = total_current_value * short_pct / 100
        rows.append([
            cat, actual_val, actual_pct,
            f"{low}% - {high}%",
            f"Short {short_pct:.2f}%",
            short_amt,
            "Increase Allocation"
        ])
        continue

    # ---- CASE 2: Within range ----
    if low <= actual_pct <= high:
        rows.append([
            cat, actual_val, actual_pct,
            f"{low}% - {high}%",
            "OK",
            0.0,
            "No Action"
        ])
        continue

    # ---- CASE 3: Excess Allocation ----
    if actual_pct > high:
        excess_pct = actual_pct - high
        excess_amt = total_current_value * excess_pct / 100
        rows.append([
            cat, actual_val, actual_pct,
            f"{low}% - {high}%",
            f"Excess {excess_pct:.2f}%",
            excess_amt,
            "Reduce Allocation"
        ])
        continue

    # ---- CASE 4: Short Allocation ----
    if actual_pct < low:
        short_pct = low - actual_pct
        short_amt = total_current_value * short_pct / 100
        rows.append([
            cat, actual_val, actual_pct,
            f"{low}% - {high}%",
            f"Short {short_pct:.2f}%",
            short_amt,
            "Increase Allocation"
        ])

# Build dataframe
alloc_table_df = pd.DataFrame(
    rows,
    columns=[
        "Category", "Actual Value (₹)", "Actual %", "Ideal Range",
        "Short / Excess %", "Short / Excess Amt (₹)", "Suggestion"
    ],
)

# Apply Indian formatting to value columns
alloc_table_df["Actual Value (₹)"] = alloc_table_df["Actual Value (₹)"].apply(inr_format)
alloc_table_df["Short / Excess Amt (₹)"] = alloc_table_df["Short / Excess Amt (₹)"].apply(inr_format)

st.dataframe(alloc_table_df, use_container_width=True)


# ------------------------------------------------------
# 5. Scheme Validation (IDCW / DIRECT / DIVIDEND)
# ------------------------------------------------------
st.markdown("### 5️⃣ Scheme Validation")

# keywords to flag (case-insensitive)
invalid_terms = ["idcw", "dividend", "direct", "payout", "pay out", "reinvestment", "regular"]

scheme_warnings = []

if scheme_col:
    for scheme in df_no_total[scheme_col].astype(str):
        s = scheme.strip()
        s_lower = s.lower()

        # ignore if clearly Reg (G)
        if "reg (g)" in s_lower or "reg(g)" in s_lower or "reg g" in s_lower:
            continue

        for term in invalid_terms:
            if term in s_lower:
                scheme_warnings.append({"scheme": s, "term": term})
                break

if not scheme_col:
    st.info("Scheme name column not detected; skipping scheme validation.")
else:
    if not scheme_warnings:
        st.markdown(
            """
            <div style='padding:10px;background:#e6ffe6;border-left:5px solid #00b300;'>
            ✅ <b>All schemes appear to be correctly tagged as Reg (G) or do not contain flagged terms.</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style='padding:8px;'><b>🚨 Scheme Plan Issues Found</b></div>
            """,
            unsafe_allow_html=True
        )
        for w in scheme_warnings:
            term_display = w["term"].upper()
            st.markdown(
                f"""
                <div style='background:#ffe6e6;padding:10px;border-left:5px solid red;margin-bottom:8px;'>
                🔴 <b>{w['scheme']}</b> — contains <b>{term_display}</b>. Please verify (recommended format: <b>Reg (G)</b>).
                </div>
                """,
                unsafe_allow_html=True
            )

# ------------------------------------------------------
# 6️⃣ Sector / Thematic / ELSS / Children Scheme Detection (Enhanced)
# ------------------------------------------------------
import re

st.markdown("### 6️⃣ Sector / Thematic / ELSS Alerts")

# --------------------------
# Keyword Groups
# --------------------------

# Children funds
children_keywords = [
    "child", "children", "childrens", "kid", "minor",
    "education", "future fund", "gift fund", "career fund"
]

# ELSS
elss_keywords = ["elss", "tax saver", "taxsaver"]

# Thematic
thematic_keywords = ["thematic", "theme", "opportunities"]

# Sector groups
sector_groups = {
    "Infrastructure": [
        "infra", "infr", "infrastr", "infrastructure", "infrastu", "infrastucture",
        "construction", "transport"
    ],
    "Banking / Financial": [
        "bank", "banking", "finance", "financial", "nbfc", "credit"
    ],
    "Technology": [
        "tech", "technology", "software", "digital", "internet", "ai", "data"
    ],
    "IT": [" it "],  # safe match
    "Pharma / Healthcare": [
        "pharma", "health", "healthcare", "biotech", "diagnostic", "life science"
    ],
    "FMCG / Consumption": [
        "consumer", "consumption", "fmcg", "retail", "food", "beverage"
    ],
    "Auto / EV": [
        "auto", "automobile", "ev", "mobility"
    ],
    "Energy / Power / Oil": [
        "energy", "power", "renewable", "solar", "wind",
        "oil", "gas", "petro"
    ],
    "Commodities / Metals": [
        "commodity", "commodities", "metal", "mining", "steel", "iron", "aluminium"
    ],
    "PSU": [
        "psu", "maharatna", "navratna", "government"
    ],
    "Global / MNC": [
        "mnc", "global", "international", "world"
    ],
    "Agriculture / Rural": [
        "agri", "agriculture", "rural", "fertilizer", "crop", "seed"
    ]
}

sector_warnings = []

# --------------------------
# Detection Loop
# --------------------------
if scheme_col:
    for scheme in df_no_total[scheme_col].astype(str):
        s = scheme.lower()

        # ---- Children Detection (Highest Priority) ----
        if any(k in s for k in children_keywords):
            sector_warnings.append(
                f"🔴 <b>{scheme}</b> — detected as <b>Children / Minor</b> fund. Please verify suitability."
            )
            continue

        # ---- ELSS ----
        if any(k in s for k in elss_keywords):
            sector_warnings.append(
                f"🔴 <b>{scheme}</b> — detected as <b>ELSS</b> fund. Please verify suitability."
            )
            continue

        # ---- Thematic ----
        if any(k in s for k in thematic_keywords):
            sector_warnings.append(
                f"🔴 <b>{scheme}</b> — detected as <b>Thematic</b> fund. Please verify suitability."
            )
            continue

        # ---- Sector Detection ----
        for sector_name, keywords in sector_groups.items():
            if any(k in s for k in keywords):
                sector_warnings.append(
                    f"🔴 <b>{scheme}</b> — detected as <b>{sector_name}</b> sector fund. Please verify suitability."
                )
                break

# --------------------------
# Display Alerts
# --------------------------
if len(sector_warnings) == 0:
    st.markdown(
        """
        <div style='padding:10px;background:#e6ffe6;border-left:5px solid #00b300;'>
        ✅ No Sectorial, Thematic, ELSS or Children schemes detected.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div style='padding:8px;'><b>🚨 Alerts Found</b></div>",
        unsafe_allow_html=True
    )

    for warn in sector_warnings:
        st.markdown(
            f"""
            <div style='background:#ffe6e6;padding:10px;border-left:5px solid red;margin-bottom:8px;'>
            {warn}
            </div>
            """,
            unsafe_allow_html=True
        )
