import streamlit as st

# Function to check if the scheme allocation is out of bounds
def check_allocation(category, allocation):
    if category == 'Small Cap' and allocation > 30:
        return 'Small Cap exceeds recommended allocation (25%-30%)'
    elif category == 'Mid Cap' and allocation > 30:
        return 'Mid Cap exceeds recommended allocation (25%-30%)'
    elif category == 'Large Cap' and allocation < 30:
        return 'Large Cap allocation is below recommended range (30%-50%)'
    elif category == 'Flexi Cap' and allocation < 30:
        return 'Flexi Cap allocation is below recommended range (30%-50%)'
    return None

# Function to check for non-Regular Growth schemes
def check_regular_growth(scheme_name):
    if 'Regular Growth' not in scheme_name:
        return f"Scheme '{scheme_name}' is not Regular Growth. Consider switching to Regular Growth."
    return None

# ---------- 4. Scheme-wise Allocation with Suggestion Box ----------

st.markdown("### 4️⃣ Scheme Allocation (Top 10 by value)")

if current_col and scheme_col:
    alloc = df_no_total[[scheme_col, current_col]].dropna()
    alloc_group = alloc.groupby(scheme_col)[current_col].sum().sort_values(ascending=False)
    total_current = df_no_total[current_col].sum()
    alloc_pct = (alloc_group / total_current * 100).round(2) if total_current > 0 else alloc_group * 0

    alloc_table = pd.DataFrame(
        {
            "Scheme": alloc_group.index.astype(str),
            "Current Value (₹)": alloc_group.values,
            "Allocation (%)": alloc_pct.values,
        }
    )

    st.dataframe(alloc_table, use_container_width=True)

    # Top 10 schemes for visualization
    top = alloc_table.head(10)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(top["Scheme"], top["Allocation (%)"])
    ax.set_xticklabels(top["Scheme"], rotation=45, ha="right")
    ax.set_ylabel("Allocation (%)")
    ax.set_title("Top 10 Schemes by Allocation")
    fig.tight_layout()
    st.pyplot(fig)

    # -------------- Suggestion Box --------------
    st.markdown("### 💡 Suggestions")

    suggestions = []
    
    # Check each scheme's allocation for criteria violation
    for idx, row in alloc_table.iterrows():
        category = row["Scheme"]
        allocation = row["Allocation (%)"]
        
        allocation_warning = check_allocation(category, allocation)
        if allocation_warning:
            suggestions.append(allocation_warning)
        
        # Check if it's a Regular Growth scheme
        regular_growth_warning = check_regular_growth(category)
        if regular_growth_warning:
            suggestions.append(regular_growth_warning)
    
    # Show suggestions (highlighting violations in red)
    if suggestions:
        for suggestion in suggestions:
            st.markdown(f"<p style='color:red'>{suggestion}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:green'>No issues detected with scheme allocations.</p>", unsafe_allow_html=True)

else:
    st.info("Could not detect Scheme / Current Value columns for scheme allocation.")
