import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import locale

# ------------------------------------------------------
# SET INDIAN NUMBER FORMAT
# ------------------------------------------------------
try:
    locale.setlocale(locale.LC_ALL, "en_IN.UTF-8")
except:
    locale.setlocale(locale.LC_ALL, "")

def inr(x):
    try:
        return "₹ " + locale.format_string("%.2f", float(x), grouping=True)
    except:
        return x


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

def find_col_like(cols, keys):
    up = [str(c).upper().strip() for c in cols]
    for k in keys:
        for orig, u in zip(cols, up):
            if k in u:
                return orig
    return None

def extract_client_name(df_header):
    for r in range(min(10, df_header.shape[0])):
        for c in range(min(5, df_header.shape[1])):
            val = str(df_header.iat[r, c])
            if val.startswith("Client:"):
                txt = val.replace("Client:", "").strip()
                if "(" in txt:
                    return txt.split("(")[0].strip()
                return txt
    return "N/A"

def extract_table(df_raw):
    for idx in range(df_raw.shape[0]):
        first = str(df_raw.iat[idx, 0]).strip().upper()
        if first in ("SCHEME NAME", "SCHEME"):
            header = df_raw.iloc[idx]
            table = df_raw.iloc[idx+1:].copy()
            table.columns = header
            table = table.dropna(how="all")
            return table, idx

    # fallback
    fne = df_raw.dropna(how="all").index[0]
    header = df_raw.iloc[fne]
    table = df_raw.iloc[fne+1:].copy()
    table.columns = header
    return table, fne

def classify_from_scheme_name(n):
    s = str(n).lower()
    if any(x in s for x in ["liquid", "overnight", "money market"]):
        return "Liquid"
    if any(x in s for x in ["gilt", "debt", "bond", "credit risk", "corporate bond"]):
        return "Debt"
    if any(x in s for x in ["hybrid", "balanced"]):
        return "Hybrid"
    if any(x in s for x in ["equity", "mid", "small", "large", "flexi", "index", "elss"]):
        return "Equity"
    return "Other"

def split_main_sub(x):
    s = str(x)
    if ":" in s:
        m, sb = s.split(":", 1)
        return m.strip(), sb.strip()
    low = s.lower()
    if "equity" in low:
        return "Equity", s
    if any(k in low for k in ["debt", "bond", "gilt"]):
        return "Debt", s
    if "liquid" in low:
        return "Liquid", s
    if "hybrid" in low:
        return "Hybrid", s
    return "Other", s

def apply_section_subcategories(df, scheme_col):
    if scheme_col is None:
        return df, False

    s = df[scheme_col].astype(str)
    mask = s.str.contains(":", regex=False) & ~s.str.contains("TOTAL", case=False)

    if mask.sum() == 0:
        return df, False

    df2 = df.copy()
    df2["SubCategory"] = s.where(mask).ffill()
    df2["MainCategory"] = df2["SubCategory"].apply(lambda x: str(x).split(":",1)[0])
    df2 = df2[~mask].copy()
    return df2, True


# ---------------------- APP UI ----------------------
st.title("📊 Portfolio Auto Analyzer")

uploaded = st.file_uploader("Upload portfolio Excel (SLA / Investwell)", type=["xlsx","xls"])

if not uploaded:
    st.info("Upload an Excel file to begin.")
    st.stop()

try:
    df_full = pd.read_excel(uploaded, sheet_name=0, header=None, dtype=str)
except:
    st.error("Cannot read Excel.")
    st.stop()

# ---------------------------------------------------------
# REMOVE ROWS LIKE:  "KAMLA DEVI (BYJPD8996A)"
# ---------------------------------------------------------
df_full = df_full[~df_full.iloc[:,0].astype(str).str.contains(r"\([A-Z0-9]{6,}\)", regex=True)]


client_name = extract_client_name(df_full)
table, header_row = extract_table(df_full)
df = table.copy()

# Detect columns
purchase_col = find_col_like(df.columns, ["PURCHASE"])
current_col = find_col_like(df.columns, ["CURRENT VALUE"])
gain_col = find_col_like(df.columns, ["GAIN"])
abs_ret_col = find_col_like(df.columns, ["ABSOLUTE"])
cagr_col = find_col_like(df.columns, ["CAGR", "XIRR"])
scheme_col = find_col_like(df.columns, ["SCHEME"])
subcat_col = find_col_like(df.columns, ["SUB CATEGORY", "TYPE"])

# Clean numeric
for col in [purchase_col, current_col, gain_col, abs_ret_col, cagr_col]:
    if col:
        df[col] = num(df[col])

# remove TOTAL rows
mask_total = df.apply(lambda r: any("TOTAL" in str(x).upper() for x in r), axis=1)
df_no_total = df[~mask_total].copy()

# Derive categories
df_no_total, used_style = apply_section_subcategories(df_no_total, scheme_col)

if not used_style:
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

# Normalize categories
norm_map = {
    "mid cap":"Mid Cap","midcap":"Mid Cap",
    "small cap":"Small Cap","smallcap":"Small Cap",
    "large cap":"Large Cap","largecap":"Large Cap",
    "flexicap":"Flexi Cap","flexi cap":"Flexi Cap"
}

def normalize(x):
    s = str(x).lower()
    for k,v in norm_map.items():
        if k in s:
            return v
    return x

df_no_total["SubCategory"] = df_no_total["SubCategory"].apply(normalize)



# ------------------------------------------------------
# 1️⃣ SUMMARY + FIXED CAGR
# ------------------------------------------------------
st.markdown("### 1️⃣ Portfolio Summary")

grand_total_cagr = None
if cagr_col:
    mask = df.iloc[:,0].astype(str).str.contains("Grand Total", case=False)
    if mask.any():
        raw = df[mask].iloc[0][cagr_col]
        grand_total_cagr = float(str(raw).replace(",","").replace("%",""))

total_purchase = df_no_total[purchase_col].sum()
total_current = df_no_total[current_col].sum()
gain = total_current - total_purchase

c1,c2,c3 = st.columns(3)
c1.metric("Client", client_name)
c2.metric("Purchase (₹)", inr(total_purchase))
c3.metric("Current (₹)", inr(total_current))

c4,c5 = st.columns(2)
c4.metric("Gain / Loss (₹)", inr(gain))
c5.metric("CAGR / XIRR (%)", f"{grand_total_cagr:.2f}" if grand_total_cagr else "N/A")

with st.expander("Scheme-level table"):
    st.dataframe(df_no_total.reset_index(drop=True))


# ------------------------------------------------------
# 2️⃣ Category Allocation
# ------------------------------------------------------
st.markdown("### 2️⃣ Category Allocation")

if current_col:
    cat_val = df_no_total.groupby("MainCategory")[current_col].sum()
    total = df_no_total[current_col].sum()
    cat_pct = (cat_val/total*100).round(2)

    df_cat = pd.DataFrame({
        "Category": cat_val.index,
        "Value": cat_val.values,
        "Allocation %": cat_pct.values
    })

    df_cat["Value"] = df_cat["Value"].apply(inr)

    st.dataframe(df_cat, use_container_width=True)

    fig, ax = plt.subplots(figsize=(2,2))
    ax.pie(cat_val.values, autopct="%1.1f%%")
    ax.set_title("Category Allocation", fontsize=9)
    st.pyplot(fig)


# ------------------------------------------------------
# 3️⃣ Sub-category Allocation
# ------------------------------------------------------
st.markdown("### 3️⃣ Sub-category Allocation")

if current_col:
    sub_val = df_no_total.groupby("SubCategory")[current_col].sum()
    total = df_no_total[current_col].sum()
    sub_pct = (sub_val/total*100).round(2)

    df_sub = pd.DataFrame({
        "SubCategory": sub_val.index,
        "Value": sub_val.values,
        "Allocation %": sub_pct.values
    })

    df_sub["Value"] = df_sub["Value"].apply(inr)
    st.dataframe(df_sub, use_container_width=True)


# ------------------------------------------------------
# 4️⃣ Scheme Allocation (All Schemes)
# ------------------------------------------------------
st.markdown("### 4️⃣ Scheme Allocation (All Schemes)")

if current_col and scheme_col:
    alloc = df_no_total[[scheme_col, current_col]]

    merged = alloc.groupby(scheme_col)[current_col].sum().sort_values(ascending=False)

    total = df_no_total[current_col].sum()
    alloc_pct = (merged/total*100)

    # RISK SCORE
    def scheme_risk(p):
        return "🟥 High Risk" if p > 10 else "🟩 Safe"

    df_scheme = pd.DataFrame({
        "Scheme": merged.index,
        "Current Value": merged.values,
        "Allocation %": alloc_pct.values,
        "Risk Score": [scheme_risk(x) for x in alloc_pct.values]
    })

    df_scheme["Current Value"] = df_scheme["Current Value"].apply(inr)

    st.dataframe(df_scheme, use_container_width=True)


# ------------------------------------------------------
# 4️⃣a AMC-wise Allocation (Table + Expanders)
# ------------------------------------------------------
st.markdown("### 4️⃣a AMC-wise Allocation")

if scheme_col:
    df_no_total["AMC"] = df_no_total[scheme_col].apply(lambda x: str(x).split()[0])

    amc_val = df_no_total.groupby("AMC")[current_col].sum()
    total = df_no_total[current_col].sum()
    amc_pct = (amc_val/total*100)

    amc_rows = []
    amc_map = {}

    for amc in amc_val.index:
        val = amc_val[amc]
        pct = amc_pct[amc]

        if pct < 20:
            risk = "🟩 Low Risk"
        elif pct <= 25:
            risk = "🟧 Medium Risk"
        else:
            risk = "🟥 High Risk"

        df_amc = df_no_total[df_no_total["AMC"] == amc][[scheme_col, current_col]]
        amc_map[amc] = df_amc

        amc_rows.append([amc, val, pct, risk, df_amc.shape[0]])

    df_amc = pd.DataFrame(amc_rows, columns=["AMC","Value","Allocation %","Risk","Schemes"])
    df_amc["Value"] = df_amc["Value"].apply(inr)

    st.dataframe(df_amc, use_container_width=True)

    st.markdown("### AMC Schemes Breakdown")

    for amc in df_amc["AMC"]:
        with st.expander(f"{amc} — Schemes"):
            temp = amc_map[amc].copy()
            temp.rename(columns={scheme_col:"Scheme", current_col:"Current Value"}, inplace=True)
            temp["Current Value"] = temp["Current Value"].apply(inr)
            st.dataframe(temp, use_container_width=True)


# ------------------------------------------------------
# 6️⃣ Sector / Thematic / ELSS Alerts
# ------------------------------------------------------
st.markdown("### 6️⃣ Sector / Thematic / ELSS Alerts")

sector_patterns = {
    "IT": r"\bit\b",
    "Technology": r"technology",
    "Tech": r"\btech\b",
    "Pharma": r"pharma",
    "Banking / Financial": r"(bank|finance|financial)",
    "Auto": r"\bauto\b",
    "Infrastructure": r"infrastructure",
    "Commodity": r"commodity",
}

sector_warnings = []

for sch in df_no_total[scheme_col].astype(str):
    s = sch.lower()

    if "elss" in s:
        sector_warnings.append(f"🔴 <b>{sch}</b> — ELSS Tax Saver")
        continue

    if "thematic" in s:
        sector_warnings.append(f"🔴 <b>{sch}</b> — Thematic Fund")
        continue

    for sec, pattern in sector_patterns.items():
        if re.search(pattern, s):
            sector_warnings.append(f"🔴 <b>{sch}</b> — Sector Fund ({sec})")
            break

if not sector_warnings:
    st.success("No Sectorial / Thematic / ELSS issues found.")
else:
    for w in sector_warnings:
        st.markdown(
            f"""
            <div style='background:#ffe6e6;padding:10px;border-left:5px solid red;margin-bottom:8px;'>
            {w}<br>Please verify suitability.
            </div>
            """,
            unsafe_allow_html=True
        )
