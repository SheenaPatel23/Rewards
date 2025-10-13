"""
Streamlit Pay Equity Regression App (Upgraded)
==============================================
- Uses 100% of the dataset for model estimation (no train/test split).
- Polished Streamlit UI with tabs, metrics, interactive Plotly charts.
- Automatic categorical handling (wraps non-numeric variables with C(...)).
- Sample data includes columns: EmployeeID, Gender, JobLevel, Grouping,
  ServiceYears, TenureinRole, Rating, Salary.
- Provides plain-English explanation of results and recommended actions.

Run:
    pip install streamlit pandas numpy matplotlib statsmodels scikit-learn plotly
    streamlit run app.py
"""

import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Helper functions
# -------------------------

def load_sample_data() -> pd.DataFrame:
    """Return a small realistic sample dataset for pay equity analysis."""
    np.random.seed(42)
    d = {
        'EmployeeID': [f'E{str(i).zfill(4)}' for i in range(1,101)],
        'Gender': np.random.choice(['Male','Female'], size=100, p=[0.55,0.45]),
        'JobLevel': np.random.choice([2,3,4,5,6], size=100, p=[0.1,0.25,0.35,0.2,0.1]),
        'Grouping': np.random.choice(['Ops','Sales','Finance','HR','IT'], size=100),
        'ServiceYears': np.round(np.random.exponential(scale=5, size=100)).astype(int),
        'TenureinRole': np.round(np.random.exponential(scale=2, size=100)).astype(int),
        'Rating': np.random.choice([1,2,3,4,5], size=100, p=[0.05,0.15,0.45,0.25,0.1]),
    }
    df = pd.DataFrame(d)

    # Construct Salary from additive model + noise
    base = 30000
    df['Salary'] = (
        base
        + df['JobLevel'] * 8000
        + df['Rating'] * 1500
        + df['ServiceYears'] * 700
        + df['Grouping'].map({'Ops':0,'Sales':4000,'Finance':2000,'HR':-500,'IT':1000})
    )
    # Introduce a small unexplained gender gap for demo purposes
    df.loc[df['Gender']=='Female', 'Salary'] *= 0.97
    # Add noise
    df['Salary'] = (df['Salary'] + np.random.normal(scale=3000, size=len(df))).round(0)
    return df

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) for UI suggestions."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical

def build_formula_auto(dep: str, indep: List[str], categorical_cols: List[str]) -> str:
    """Construct a formula string for statsmodels, automatically wrapping categorical vars."""
    rhs_terms = []
    for v in indep:
        if v in categorical_cols:
            rhs_terms.append(f"C({v})")
        else:
            rhs_terms.append(v)
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"
    return f"{dep} ~ {rhs}"

def run_ols_formula(formula: str, df: pd.DataFrame):
    """Fit OLS using statsmodels and return the results object."""
    model = smf.ols(formula=formula, data=df).fit()
    return model

def to_download_link(df: pd.DataFrame, filename: str) -> str:
    """Return a data URI for browser download."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"

# -------------------------
# App layout
# -------------------------

st.set_page_config(page_title="Rewards — Pay Equity Regression", layout="wide")
st.title("Rewards — Pay Equity Regression Dashboard")
st.markdown(
    "Professional, interactive dashboard to explore pay equity using OLS regression. "
    "Upload your employee-level CSV or use the built-in sample dataset."
)

# Sidebar: data loading and model controls
st.sidebar.header("1) Data")
upload = st.sidebar.file_uploader("Upload employee CSV", type=["csv"])
use_sample = st.sidebar.button("Load sample dataset") if upload is None else False

if upload is not None:
    try:
        raw_df = pd.read_csv(upload)
        st.sidebar.success("CSV loaded")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.stop()
elif use_sample or upload is None:
    raw_df = load_sample_data()
    st.sidebar.info("Using built-in sample dataset")

# Validate expected columns presence for UX (not required)
expected = ['EmployeeID','Gender','JobLevel','Grouping','ServiceYears','TenureinRole','Rating','Salary']
present = [c for c in expected if c in raw_df.columns]
missing = [c for c in expected if c not in raw_df.columns]

# Show primary dataset info in sidebar
st.sidebar.markdown("**Dataset summary**")
st.sidebar.write(f"Rows: {raw_df.shape[0]} — Columns: {raw_df.shape[1]}")
if missing:
    st.sidebar.warning(f"Missing example columns (not required): {', '.join(missing)}")

numeric_cols, categorical_cols = detect_column_types(raw_df)

# Model configuration area
st.sidebar.header("2) Model configuration")
# Default dependent variable: Salary if present, else first numeric
if 'Salary' in numeric_cols:
    default_dep = 'Salary'
else:
    default_dep = numeric_cols[0] if numeric_cols else None

dep_var = st.sidebar.selectbox("Dependent variable (target)", options=numeric_cols, index=numeric_cols.index(default_dep) if default_dep in numeric_cols else 0)

# Default independent variables
default_indep = [c for c in ['JobLevel','Rating','ServiceYears','TenureinRole','Gender','Grouping'] if c in raw_df.columns]
indep_vars = st.sidebar.multiselect("Independent variables (features)", options=[*numeric_cols, *categorical_cols], default=default_indep)

# Allow the user to edit the formula directly or auto-generate
auto_formula = build_formula_auto(dep_var, indep_vars, categorical_cols)
formula = st.sidebar.text_area("Regression formula (editable)", value=auto_formula, height=120,
                               help="You can edit the formula. C(var) wraps categorical variables. Example: Salary ~ JobLevel + C(Gender) + C(Grouping)")

# Run button in sidebar
run_model = st.sidebar.button("Run regression (use 100% data)")

# Tabs for main content
tabs = st.tabs(["Data", "Model", "Results", "Insights & Actions"])

# -------------------------
# DATA TAB
# -------------------------
with tabs[0]:
    st.header("Dataset — preview (interactive)")
    st.markdown("Full dataset shown below (use table controls to sort/filter).")
    st.dataframe(raw_df, use_container_width=True)

    with st.expander("Dataset summary statistics"):
        st.write(raw_df.describe(include='all').T)

# If user didn't click run, show helpful note
if not run_model:
    with tabs[1]:
        st.info("Configure model on the left and click **Run regression (use 100% data)** to estimate using the full dataset.")
    st.stop()

# -------------------------
# MODEL TAB (results computed)
# -------------------------
# Run regression on full dataset (100%)
try:
    model = run_ols_formula(formula, raw_df)
except Exception as e:
    st.error(f"Error fitting model: {e}")
    st.stop()

# Prepare coefficient table
coef = model.params.reset_index()
coef.columns = ['term', 'coef']
coef['std_err'] = model.bse.values
coef['t'] = model.tvalues.values
coef['p_value'] = model.pvalues.values
conf_int = model.conf_int()
coef['ci_lower'] = conf_int.loc[:,0].values
coef['ci_upper'] = conf_int.loc[:,1].values

# Predicted and residuals added to dataset copy
results_df = raw_df.copy()
results_df['Predicted'] = model.predict(results_df)
results_df['Residual'] = results_df[dep_var] - results_df['Predicted']

# MODEL TAB content
with tabs[1]:
    st.header("Model overview (100% of data)")
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared", f"{model.rsquared:.3f}")
    col2.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
    col3.metric("Observations (N)", f"{int(model.nobs)}")
    col4.metric(f"Mean {dep_var}", f"{results_df[dep_var].mean():.0f}")

    st.markdown("**Model summary (compact)**")
    st.text(model.summary().as_text())

    st.markdown("---")
    st.subheader("Coefficients")
    st.markdown("Coefficients with standard errors, t-statistics, p-values and 95% CI.")
    st.dataframe(coef.sort_values('p_value').reset_index(drop=True), use_container_width=True)

# -------------------------
# RESULTS TAB
# -------------------------
with tabs[2]:
    st.header("Interactive results & diagnostics")

    # Coefficient bar chart (exclude intercept for clarity)
    coef_plot_df = coef[coef['term'] != 'Intercept'].copy()
    if coef_plot_df.empty:
        st.info("No coefficients to plot (maybe only intercept).")
    else:
        fig_coef = px.bar(
            coef_plot_df,
            x='coef',
            y='term',
            orientation='h',
            error_x=coef_plot_df['std_err'] * 2,
            title="Regression coefficients (95% CI shown as error bars)",
            hover_data=['std_err', 't', 'p_value', 'ci_lower', 'ci_upper']
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("### Predicted vs Actual")
    fig_pred = px.scatter(
        results_df,
        x='Predicted',
        y=dep_var,
        hover_data=['EmployeeID'] if 'EmployeeID' in results_df.columns else None,
        title="Predicted vs Actual",
        trendline="ols"
    )
    # Add 45-degree line to help judge fit
    minv = min(results_df[['Predicted', dep_var]].min().min(), 0)
    maxv = max(results_df[['Predicted', dep_var]].max().max(), 0)
    fig_pred.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode='lines', line=dict(dash='dash'), showlegend=False))
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("### Residuals diagnostics")
    fig_res_hist = px.histogram(results_df, x='Residual', nbins=30, title="Residuals distribution")
    st.plotly_chart(fig_res_hist, use_container_width=True)

    fig_res_scatter = px.scatter(results_df, x='Predicted', y='Residual',
                                 hover_data=['EmployeeID'] if 'EmployeeID' in results_df.columns else None,
                                 title="Residuals vs Predicted")
    st.plotly_chart(fig_res_scatter, use_container_width=True)

    st.markdown("### Employees with largest negative residuals (underpaid vs model) — top 10")
    top_underpaid = results_df.sort_values('Residual').head(10)
    display_cols = ['EmployeeID'] + [dep_var, 'Predicted', 'Residual'] if 'EmployeeID' in results_df.columns else [dep_var, 'Predicted', 'Residual']
    st.dataframe(top_underpaid[display_cols], use_container_width=True)

    st.markdown("### Employees with largest positive residuals (overpaid vs model) — top 10")
    top_overpaid = results_df.sort_values('Residual', ascending=False).head(10)
    st.dataframe(top_overpaid[display_cols], use_container_width=True)

    # Downloads
    st.markdown("---")
    st.markdown("**Download outputs**")
    coeff_href = to_download_link(coef, "regression_coefficients.csv")
    st.markdown(f"[Download coefficients CSV]({coeff_href})")
    results_href = to_download_link(results_df, "employee_regression_results.csv")
    st.markdown(f"[Download employee-level predictions CSV]({results_href})")

# -------------------------
# INSIGHTS & ACTIONS TAB
# -------------------------
with tabs[3]:
    st.header("Interpretation, insights and recommended actions")
    st.markdown(
        "This section provides an automated, plain-English interpretation of the fitted model "
        "and practical actions for HR/Rewards teams. Use these as starting points; always combine "
        "with business context and manual review."
    )

    # Short auto-interpretation: significant variables (p < 0.05)
    sig_thresh = 0.05
    significant = coef[coef['p_value'] < sig_thresh].copy()
    # Exclude Intercept for narrative
    significant_no_intercept = significant[significant['term'] != 'Intercept']

    with st.expander("Model interpretation (automated)"):
        st.subheader("Key numbers")
        st.write(f"- R-squared: **{model.rsquared:.3f}** — proportion of variance in **{dep_var}** explained by the model.")
        st.write(f"- Observations: **{int(model.nobs)}**")
        st.write(f"- Average {dep_var}: **{results_df[dep_var].mean():.0f}**")

        if significant_no_intercept.empty:
            st.write("- No explanatory variables reach p < 0.05 significance at the default threshold.")
        else:
            st.write("- **Significant predictors (p < 0.05):**")
            for _, row in significant_no_intercept.sort_values('p_value').iterrows():
                term = row['term']
                coef_val = row['coef']
                pval = row['p_value']
                # Plain-language for categorical indicator terms like C(Gender)[T.Female]
                if term.startswith("C("):
                    # e.g. C(Gender)[T.Female] -> Gender = Female
                    try:
                        inside, catpart = term.split("]", 1)[0].split("[T.")
                        var = inside[2:-1]
                        level = catpart
                        sign = "increase" if coef_val > 0 else "decrease"
                        st.write(f"  - **{var} = {level}**: estimated {sign} of **{coef_val:.0f}** on {dep_var} (p={pval:.3f}).")
                    except Exception:
                        st.write(f"  - **{term}**: coef={coef_val:.2f} (p={pval:.3f})")
                else:
                    sign = "increase" if coef_val > 0 else "decrease"
                    st.write(f"  - **{term}**: one unit {sign} associated with **{coef_val:.0f}** change in {dep_var} (p={pval:.3f}).")

        st.markdown("---")
        st.subheader("How to read these results (brief)")
        st.markdown(
            """
            - The model estimates *associations*, not causal effects. For example, a positive coefficient for JobLevel
              indicates higher JobLevel is associated with higher Salary after controlling for other included factors.
            - Coefficients for categorical variables (e.g. `C(Gender)[T.Female]`) represent the *average difference*
              between that category and the reference category, holding other variables constant.
            - Look at p-values for statistical evidence; also examine effect sizes (coef) relative to mean salary to judge business significance.
            - Residuals show individual employees who are paid more or less than the model predicts — these are good candidates for manual review.
            """
        )

    with st.expander("Suggested actions (practical checklist)"):
        st.subheader("Immediate actions")
        st.markdown(
            """
            1. **Investigate significant differences**: For any protected characteristic (e.g. Gender) showing a statistically
               significant coefficient, run subgroup checks (e.g., within JobLevel or Grouping) to confirm whether the gap persists.
            2. **Manual review for outliers**: Use the residuals table to identify employees with large negative residuals (potential underpaid)
               and large positive residuals (potential overpaid). Check performance, role, tenure, and any special allowance.
            3. **Data quality checks**: Verify fields like JobLevel, ServiceYears, Salary for input errors or inconsistent units.
            4. **Adjust policy or pay actions**: If unexplained, material gaps exist after review, consider targeted pay adjustments,
               salary band updates, and/or transparent pay criteria documentation.
            5. **Repeat periodically**: Re-run the analysis after corrective actions and as new hires/promotions occur.
            """
        )

        st.subheader("Further analysis recommendations")
        st.markdown(
            """
            - **Stratified models**: Run separate regressions within JobLevel bands or Groupings to detect localized gaps.
            - **Add controls**: If available, include variables such as education, certifications, or performance calibration score.
            - **Non-linear effects**: Consider adding polynomial terms for tenure if the relationship is not linear.
            - **Equity review board**: For substantive findings, convene cross-functional review with HR, legal, and business leaders.
            """
        )

    with st.expander("Limitations & next steps"):
        st.write(
            """
            - This OLS model is a diagnostic tool — it does not prove discrimination. Use it to surface patterns for investigation.
            - Model quality depends on the variables included. Missing important covariates can bias coefficient interpretation.
            - Consider pairing this quantitative analysis with qualitative review (manager justification, role descriptions).
            """
        )

# Footer
st.markdown("---")
st.caption("Built for the Rewards team — Finance Data & Analytics  •  Use responsibly: combine statistical output with business judgment.")
