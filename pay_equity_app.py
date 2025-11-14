

import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# -------------------------
# Brand colors (V.Group-like)
# -------------------------
NAVY = "#002B45"
TEAL = "#00A3E0"
LIME = "#68DA6A"
LIGHT_GREY = "#F5F5F5"
WHITE = "#FFFFFF"

# -------------------------
# Helper functions
# -------------------------
def load_sample_data() -> pd.DataFrame:
    """Return sample dataset (only for quick testing). You will upload real CSV in production."""
    np.random.seed(42)
    df = pd.DataFrame({
        'EmployeeID': [f'E{str(i).zfill(4)}' for i in range(1, 101)],
        'Gender': np.random.choice(['Male', 'Female'], size=100, p=[0.55, 0.45]),
        'JobLevel': np.random.choice([2, 3, 4, 5, 6], size=100),
        'Grouping': np.random.choice(['Ops', 'Sales', 'Finance', 'HR', 'IT'], size=100),
        'Grouping Category': np.random.choice(['Management', 'Support', 'Operational'], size=100),
        'ServiceYears': np.round(np.random.exponential(scale=5, size=100)).astype(int),
        'TenureinRole': np.round(np.random.exponential(scale=2, size=100)).astype(int),
        'Rating': np.random.choice([1, 2, 3, 4, 5], size=100)
    })
    base = 30000
    df['Salary'] = (
        base
        + df['JobLevel'] * 8000
        + df['Rating'] * 1500
        + df['ServiceYears'] * 700
        + df['Grouping'].map({'Ops': 0, 'Sales': 4000, 'Finance': 2000, 'HR': -500, 'IT': 1000})
    )
    df.loc[df['Gender'] == 'Female', 'Salary'] *= 0.97
    df['Salary'] = (df['Salary'] + np.random.normal(scale=3000, size=len(df))).round(0)
    return df

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical

def build_formula_auto(dep: str, indep: List[str], categorical_cols: List[str]) -> str:
    """Auto-wrap categorical variables using C(...)."""
    rhs_terms = []
    for v in indep:
        if v in categorical_cols:
            rhs_terms.append(f"C({v})")
        else:
            rhs_terms.append(v)
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"
    return f"{dep} ~ {rhs}"

def run_ols_formula(formula: str, df: pd.DataFrame):
    """Fit OLS and return fitted model."""
    model = smf.ols(formula=formula, data=df).fit()
    return model

def to_download_link(df: pd.DataFrame, filename: str) -> str:
    """Return data URI for download links in Streamlit."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"

def format_number(val):
    if pd.isna(val):
        return ""
    try:
        return f"{val:,.0f}"
    except Exception:
        return str(val)

def model_summary_table(model) -> pd.DataFrame:
    """Return a one-row table with key model metrics formatted."""
    stats = {
        'R-squared': model.rsquared,
        'Adj. R-squared': model.rsquared_adj,
        'F-statistic': model.fvalue,
        'Prob (F-stat)': model.f_pvalue,
        'Observations': int(model.nobs),
        'AIC': model.aic,
        'BIC': model.bic
    }
    df = pd.DataFrame([stats])
    # Format numbers for display (keep numeric behind the scenes if needed)
    df_display = df.copy()
    df_display['R-squared'] = df_display['R-squared'].map(lambda x: f"{x:.3f}")
    df_display['Adj. R-squared'] = df_display['Adj. R-squared'].map(lambda x: f"{x:.3f}")
    df_display['F-statistic'] = df_display['F-statistic'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    df_display['Prob (F-stat)'] = df_display['Prob (F-stat)'].map(lambda x: f"{x:.3g}" if pd.notna(x) else "")
    df_display['Observations'] = df_display['Observations'].map(lambda x: f"{x:,}")
    df_display['AIC'] = df_display['AIC'].map(lambda x: f"{x:.1f}")
    df_display['BIC'] = df_display['BIC'].map(lambda x: f"{x:.1f}")
    return df_display

# -------------------------
# App layout & controls
# -------------------------
st.set_page_config(page_title="Rewards — Pay Equity Regression", layout="wide")
st.markdown(f"<h1 style='color:{NAVY}; margin-bottom:0px'>Rewards — Pay Equity Regression</h1>", unsafe_allow_html=True)
st.markdown(f"<div style='color:{TEAL}; margin-top:4px;'>Interactive, explainable OLS for pay equity — upload your CSV to begin.</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar: Data + Model controls
st.sidebar.header("Data")
upload = st.sidebar.file_uploader("Upload employee CSV (must include Salary)", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (for testing)", value=False)

if upload is not None:
    try:
        raw_df = pd.read_csv(upload)
        st.sidebar.success("CSV loaded")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.stop()
elif use_sample:
    raw_df = load_sample_data()
    st.sidebar.info("Loaded sample dataset")
else:
    st.sidebar.info("Upload a CSV or enable 'Use sample dataset' to continue")
    st.stop()

# Validate required columns: ignore "Total Grouping" if present
required = ['EmployeeID','Gender','JobLevel','Grouping','Grouping Category','ServiceYears','TenureinRole','Rating','Salary']
missing = [c for c in required if c not in raw_df.columns]
if missing:
    st.error(f"Missing required columns in uploaded CSV: {', '.join(missing)}. Please include them (ignore 'Total Grouping').")
    st.stop()

# Keep only relevant columns (ignore 'Total Grouping' if present)
cols_to_keep = [c for c in required if c in raw_df.columns]
raw_df = raw_df[cols_to_keep].copy()

numeric_cols, categorical_cols = detect_column_types(raw_df)

st.sidebar.header("Model configuration")
# dependent variable (Salary expected)
dep_default = 'Salary' if 'Salary' in numeric_cols else (numeric_cols[0] if numeric_cols else None)
dep_var = st.sidebar.selectbox("Dependent variable", options=numeric_cols, index=numeric_cols.index(dep_default) if dep_default in numeric_cols else 0)

# Suggested independent variables, ensure categorical grouping fields included
default_indep = [c for c in ['JobLevel','Rating','ServiceYears','TenureinRole','Gender','Grouping','Grouping Category'] if c in raw_df.columns and c != dep_var]
indep_vars = st.sidebar.multiselect("Independent variables", options=[*numeric_cols,*categorical_cols], default=default_indep)

# Auto-wrap categorical: force Gender, Grouping, Grouping Category to be treated as categorical if present
forced_categoricals = [c for c in ['Gender','Grouping','Grouping Category'] if c in raw_df.columns]
# Display checkbox to force full-data (no test split)
use_full_data = st.sidebar.checkbox("Use 100% data (no test split)", value=False)
test_size_pct = st.sidebar.slider("Test set size (%) when not using full data", min_value=0, max_value=50, value=20, step=5)

# Build formula auto and editable
formula_auto = build_formula_auto(dep_var, indep_vars, categorical_cols)
# Ensure forced categoricals are wrapped if present and not already wrapped
# We'll include them automatically in the formula if they are in indep_vars
formula = st.sidebar.text_area("Regression formula (editable)", value=formula_auto, height=140, help="Categorical variables should be wrapped as C(var). You can edit the formula manually.")

# Run button
run_model = st.sidebar.button("Run regression")

# Tabs (with clear labels / icons)
tabs = st.tabs(["📊 Data", "📈 Model", "📉 Results", "💡 Insights"])

# -------------------------
# DATA TAB
# -------------------------
with tabs[0]:
    st.subheader("Dataset")
    st.write(f"Rows: **{raw_df.shape[0]:,}** — Columns: **{raw_df.shape[1]}**")
    st.dataframe(raw_df, use_container_width=True)
    with st.expander("Summary statistics"):
        summary = raw_df.describe(include='all').T
        # Format numeric columns with thousand separators
        def fmt(x):
            if isinstance(x, (int, float, np.generic)) and not pd.isna(x):
                return format_number(x)
            return x
        st.dataframe(summary.applymap(lambda x: fmt(x)))

if not run_model:
    with tabs[1]:
        st.info("Configure the model in the sidebar and click **Run regression**.")
    st.stop()

# -------------------------
# Fit model (train/test logic)
# -------------------------
# Prepare dataset copy (drop na rows for selected vars)
model_df = raw_df.dropna(subset=[dep_var] + indep_vars).copy()

# If user didn't include forced categoricals in indep_vars but they exist, optionally include them
# (we won't add them silently — respecting user's selected indep_vars)

# If not using full data, split
if not use_full_data and test_size_pct > 0:
    train_df, test_df = train_test_split(model_df, test_size=test_size_pct/100, random_state=42)
else:
    train_df = model_df.copy()
    test_df = pd.DataFrame(columns=model_df.columns)

# Ensure formula wraps the forced categoricals if they appear in indep_vars and not already wrapped
# Simple heuristic: replace occurrences of variable name with C(var) if it's in categorical list and not already 'C('
def ensure_categorical_wrapping(formula_text, categorical_list):
    # Do a safe replacement: only replace standalone variable names on RHS
    lhs, rhs = formula_text.split("~")
    rhs = rhs.strip()
    terms = [t.strip() for t in rhs.split("+")]
    new_terms = []
    for t in terms:
        raw_t = t
        # if term already contains C( or is an expression, keep as-is
        if ("C(" in t) or (":" in t) or ("*" in t) or ("(" in t and ")" in t and t.strip().split("(")[0] != "C"):
            new_terms.append(t)
            continue
        # strip possible function wrappers/spaces
        varname = t.replace(" ", "")
        if varname in categorical_list:
            new_terms.append(f"C({varname})")
        else:
            new_terms.append(t)
    rhs_new = " + ".join(new_terms)
    return f"{lhs.strip()} ~ {rhs_new}"

formula_checked = ensure_categorical_wrapping(formula, forced_categoricals)

# Fit model on training set (or full set if use_full_data)
try:
    model = run_ols_formula(formula_checked, train_df)
except Exception as e:
    st.error(f"Error fitting model: {e}")
    st.stop()

# Prepare coefficient table
coef = model.params.reset_index()
coef.columns = ['term','coef']
coef['std_err'] = model.bse.values
coef['t'] = model.tvalues.values
coef['p_value'] = model.pvalues.values
conf_int = model.conf_int()
coef['ci_lower'] = conf_int.iloc[:,0].values
coef['ci_upper'] = conf_int.iloc[:,1].values

# Prepare predictions & residuals
train_results = train_df.copy()
train_results['Predicted'] = model.predict(train_results)
train_results['Residual'] = train_results[dep_var] - train_results['Predicted']

if not test_df.empty:
    test_results = test_df.copy()
    test_results['Predicted'] = model.predict(test_results)
    test_results['Residual'] = test_results[dep_var] - test_results['Predicted']
else:
    test_results = pd.DataFrame(columns=train_results.columns)

# Combined results for download/display
combined_results = pd.concat([train_results, test_results], axis=0, ignore_index=True, sort=False)

# -------------------------
# MODEL TAB (metrics + coeffs table)
# -------------------------
with tabs[1]:
    st.subheader("Model summary")
    sum_table = model_summary_table(model)
    st.table(sum_table)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared", f"{model.rsquared:.3f}")
    col2.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
    col3.metric("Observations (train)", f"{int(model.nobs):,}")
    col4.metric("Mean Salary", f"{train_results[dep_var].mean():,.0f}")

    st.markdown("**Regression coefficients**")
    # Show coefficients in a friendly table (formatted)
    coef_display = coef.copy()
    # Format numeric columns for display but keep numeric behind the scenes
    coef_display['coef_display'] = coef_display['coef'].map(lambda x: f"{x:,.0f}")
    coef_display['std_err_display'] = coef_display['std_err'].map(lambda x: f"{x:,.2f}")
    coef_display['t_display'] = coef_display['t'].map(lambda x: f"{x:,.2f}")
    coef_display['p_value_display'] = coef_display['p_value'].map(lambda x: f"{x:.3g}")
    coef_display['ci_display'] = coef_display.apply(lambda r: f"[{r['ci_lower']:.0f}, {r['ci_upper']:.0f}]", axis=1)

    coef_table = coef_display[['term','coef_display','std_err_display','t_display','p_value_display','ci_display']].rename(columns={
        'coef_display':'Coefficient',
        'std_err_display':'Std. Err',
        't_display':'t',
        'p_value_display':'p-value',
        'ci_display':'95% CI'
    })
    st.dataframe(coef_table, use_container_width=True)

    # Download coefficients
    coeff_href = to_download_link(coef[['term','coef','std_err','t','p_value','ci_lower','ci_upper']], 'regression_coefficients.csv')
    st.markdown(f"[Download coefficients CSV]({coeff_href})", unsafe_allow_html=True)

# -------------------------
# RESULTS TAB (interactive charts)
# -------------------------
with tabs[2]:
    st.subheader("Predicted vs Actual (interactive)")

    # For the chart: actual points (navy), predicted line (teal).
    # Use a sorted x-range so the predicted line is smooth
    preds_sorted = combined_results.sort_values('Predicted')
    fig = go.Figure()
    # Actual points
    fig.add_trace(go.Scatter(
        x=combined_results['Predicted'],
        y=combined_results[dep_var],
        mode='markers',
        marker=dict(color=NAVY, size=7),
        name='Actual',
        hovertemplate=(
            "Employee: %{customdata[0]}<br>Predicted: %{x:,.0f}<br>Actual: %{y:,.0f}<br>Residual: %{customdata[1]:,.0f}"
        ),
        customdata=np.stack([combined_results.get('EmployeeID', pd.Series(['']*len(combined_results))).values,
                             combined_results['Residual'].fillna(0).values], axis=1)
    ))
    # Predicted = x (45-degree) line: plot predicted vs predicted
    fig.add_trace(go.Scatter(
        x=preds_sorted['Predicted'],
        y=preds_sorted['Predicted'],
        mode='lines',
        line=dict(color=TEAL, width=2, dash='dash'),
        name='Perfect prediction'
    ))

    fig.update_layout(
        xaxis_title="Predicted Salary",
        yaxis_title="Actual Salary",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residual diagnostics
    st.subheader("Residual diagnostics")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(combined_results, x='Residual', nbins=30, title="Residuals distribution", marginal="box")
        fig_hist.update_layout(template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        fig_res = px.scatter(combined_results, x='Predicted', y='Residual', hover_data=['EmployeeID', dep_var],
                             title="Residuals vs Predicted")
        fig_res.update_traces(marker=dict(color=NAVY))
        fig_res.update_layout(template="plotly_white")
        st.plotly_chart(fig_res, use_container_width=True)

    # Coefficients bar (exclude intercept)
    st.subheader("Coefficients (visual)")
    plot_coef_df = coef[coef['term'] != 'Intercept'].copy()
    if not plot_coef_df.empty:
        plot_coef_df = plot_coef_df.reset_index(drop=True)
        plot_coef_df['error'] = plot_coef_df['std_err'] * 2
        plot_coef_df['coef_float'] = plot_coef_df['coef'].astype(float)
        fig_coef = px.bar(
            plot_coef_df,
            x='coef_float',
            y='term',
            orientation='h',
            error_x='error',
            title="Regression coefficients (95% CI)",
            labels={'coef_float':'Coefficient','term':'Variable'},
            color_discrete_sequence=[TEAL]
        )
        fig_coef.update_layout(template="plotly_white")
        st.plotly_chart(fig_coef, use_container_width=True)
    else:
        st.info("No coefficients to display (only intercept or no variables).")

    # Top residuals tables
    st.markdown("### Employees: largest negative residuals (potentially underpaid)")
    under = combined_results.sort_values('Residual').head(10)
    show_cols = ['EmployeeID', dep_var, 'Predicted', 'Residual']
    st.dataframe(under[show_cols].assign(**{
        dep_var: under[dep_var].map(lambda x: f"{x:,.0f}"),
        'Predicted': under['Predicted'].map(lambda x: f"{x:,.0f}"),
        'Residual': under['Residual'].map(lambda x: f"{x:,.0f}")
    }), use_container_width=True)

    st.markdown("### Employees: largest positive residuals (potentially overpaid)")
    over = combined_results.sort_values('Residual', ascending=False).head(10)
    st.dataframe(over[show_cols].assign(**{
        dep_var: over[dep_var].map(lambda x: f"{x:,.0f}"),
        'Predicted': over['Predicted'].map(lambda x: f"{x:,.0f}"),
        'Residual': over['Residual'].map(lambda x: f"{x:,.0f}")
    }), use_container_width=True)

    # Download combined results
    results_href = to_download_link(combined_results, 'employee_regression_results.csv')
    st.markdown(f"[Download employee-level predictions & residuals CSV]({results_href})", unsafe_allow_html=True)

    # Also let user download a presentation-ready simplified CSV with key columns
    simplified = combined_results[['EmployeeID','Predicted','Residual']].copy()
    simplified['Predicted'] = simplified['Predicted'].map(lambda x: f"{x:,.0f}")
    simplified['Residual'] = simplified['Residual'].map(lambda x: f"{x:,.0f}")
    simple_href = to_download_link(simplified, 'predicted_residuals_simple.csv')
    st.markdown(f"[Download simple predictions CSV]({simple_href})", unsafe_allow_html=True)

# -------------------------
# INSIGHTS TAB
# -------------------------
with tabs[3]:
    st.subheader("Automated interpretation & recommended actions")
    st.markdown(f"- **R-squared**: {model.rsquared:.3f} — proportion of variance in **{dep_var}** explained by the model.")
    st.markdown(f"- **Observations used (train)**: {int(model.nobs):,}")
    st.markdown(f"- **Mean {dep_var} (train)**: {train_results[dep_var].mean():,.0f}")

    # Significant predictors
    sig = coef[(coef['p_value'] < 0.05) & (coef['term'] != 'Intercept')].copy()
    if sig.empty:
        st.write("No predictors reach p < 0.05 at the chosen model specification.")
    else:
        st.markdown("**Significant predictors (p < 0.05)**")
        # Present in readable sentences
        for _, r in sig.sort_values('p_value').iterrows():
            term = r['term']
            coef_val = r['coef']
            pval = r['p_value']
            # Try to parse categorical term like C(Gender)[T.Female]
            if isinstance(term, str) and term.startswith("C(") and "[T." in term:
                try:
                    inside = term.split("]")[0]  # C(Gender)[T.Female
                    var = inside.split("(")[1].split(")")[0]  # Gender
                    level = inside.split("[T.")[1]  # Female
                    direction = "higher" if coef_val > 0 else "lower"
                    st.write(f"- **{var} = {level}** associated with {direction} {dep_var} of **{coef_val:,.0f}** (p={pval:.3g}).")
                except Exception:
                    st.write(f"- {term}: coef={coef_val:,.0f}, p={pval:.3g}")
            else:
                # numeric variable
                st.write(f"- **{term}**: one unit change associated with **{coef_val:,.0f}** change in {dep_var} (p={pval:.3g}).")

    with st.expander("Recommended immediate actions"):
        st.markdown("""
        1. Investigate any statistically significant differences found for **Gender**, **Grouping**, or **Grouping Category**.
        2. Use the Residuals table to identify individuals for manual review (e.g., underpaid employees with large negative residuals).
        3. Check data quality (units, missing values) for Salary, JobLevel, ServiceYears, Rating.
        4. Consider stratified models (run the same regression within JobLevel bands or Grouping to check localized gaps).
        5. If substantive unexplained gaps remain after review, prepare targeted pay adjustments and governance review.
        """)

    with st.expander("Limitations"):
        st.markdown("""
        - This is an associative OLS model; coefficients show relationships, not proof of causality.
        - Omitted variables (education, certifications, location allowances) can bias estimates.
        - Use quantitative outputs together with qualitative manager justification and role review.
        """)

# Footer
st.markdown("---")
st.caption("Built for the Rewards team — Finance Data & Analytics  •  Use results alongside business context.")
