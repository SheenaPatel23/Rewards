"""
Streamlit Pay Equity Regression App — V.Group Branded
====================================================
- Uses 100% of dataset for regression.
- Advanced tab layout with highlighted tabs.
- Interactive Plotly charts (actual vs predicted, residuals, coefficients).
- V.Group brand colors applied.
- Numbers formatted with thousand separators.
- Model summary displayed in clean table.
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
# Brand colors (from V.Group website)
# -------------------------
NAVY = "#002B45"
TEAL = "#00A3E0"
LIGHT_GREY = "#F5F5F5"
WHITE = "#FFFFFF"

# -------------------------
# Helper functions
# -------------------------

def load_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    d = {
        'EmployeeID': [f'E{str(i).zfill(4)}' for i in range(1,101)],
        'Gender': np.random.choice(['Male','Female'], size=100, p=[0.55,0.45]),
        'JobLevel': np.random.choice([2,3,4,5,6], size=100),
        'Grouping': np.random.choice(['Ops','Sales','Finance','HR','IT'], size=100),
        'ServiceYears': np.round(np.random.exponential(scale=5, size=100)).astype(int),
        'TenureinRole': np.round(np.random.exponential(scale=2, size=100)).astype(int),
        'Rating': np.random.choice([1,2,3,4,5], size=100),
    }
    df = pd.DataFrame(d)
    base = 30000
    df['Salary'] = (
        base
        + df['JobLevel'] * 8000
        + df['Rating'] * 1500
        + df['ServiceYears'] * 700
        + df['Grouping'].map({'Ops':0,'Sales':4000,'Finance':2000,'HR':-500,'IT':1000})
    )
    df.loc[df['Gender']=='Female', 'Salary'] *= 0.97
    df['Salary'] = (df['Salary'] + np.random.normal(scale=3000, size=len(df))).round(0)
    return df

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical

def build_formula_auto(dep: str, indep: List[str], categorical_cols: List[str]) -> str:
    rhs_terms = []
    for v in indep:
        if v in categorical_cols:
            rhs_terms.append(f"C({v})")
        else:
            rhs_terms.append(v)
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"
    return f"{dep} ~ {rhs}"

def run_ols_formula(formula: str, df: pd.DataFrame):
    return smf.ols(formula=formula, data=df).fit()

def to_download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"

def format_number(val):
    if pd.isna(val):
        return ""
    return f"{val:,.0f}"

# -------------------------
# App layout
# -------------------------
st.set_page_config(page_title="Rewards — Pay Equity Regression", layout="wide")
st.markdown(
    f"<h1 style='color:{NAVY}'>Rewards — Pay Equity Regression Dashboard</h1>",
    unsafe_allow_html=True
)

# Sidebar controls
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

numeric_cols, categorical_cols = detect_column_types(raw_df)

st.sidebar.header("2) Model configuration")
dep_var = st.sidebar.selectbox("Dependent variable (target)", options=numeric_cols, index=numeric_cols.index("Salary") if "Salary" in numeric_cols else 0)
default_indep = [c for c in ["JobLevel","Rating","ServiceYears","TenureinRole","Gender","Grouping"] if c in raw_df.columns]
indep_vars = st.sidebar.multiselect("Independent variables (features)", options=[*numeric_cols, *categorical_cols], default=default_indep)
auto_formula = build_formula_auto(dep_var, indep_vars, categorical_cols)
formula = st.sidebar.text_area("Regression formula (editable)", value=auto_formula, height=120)
run_model = st.sidebar.button("Run regression (use 100% data)")

tabs = st.tabs([
    f"<span style='color:{TEAL}; font-weight:bold'>Data</span>",
    f"<span style='color:{TEAL}; font-weight:bold'>Model</span>",
    f"<span style='color:{TEAL}; font-weight:bold'>Results</span>",
    f"<span style='color:{TEAL}; font-weight:bold'>Insights</span>"
])

# -------------------------
# DATA TAB
# -------------------------
with tabs[0]:
    st.header("Dataset preview")
    st.dataframe(raw_df, use_container_width=True)
    with st.expander("Summary statistics"):
        st.dataframe(raw_df.describe(include='all').T.applymap(lambda x: format_number(x) if isinstance(x,(int,float)) else x))

if not run_model:
    with tabs[1]:
        st.info("Configure model and click **Run regression** to compute results.")
    st.stop()

# -------------------------
# MODEL TAB
# -------------------------
model = run_ols_formula(formula, raw_df)

coef = model.params.reset_index()
coef.columns = ['term','coef']
coef['std_err'] = model.bse.values
coef['t'] = model.tvalues.values
coef['p_value'] = model.pvalues.values
conf_int = model.conf_int()
coef['ci_lower'] = conf_int[0].values
coef['ci_upper'] = conf_int[1].values

results_df = raw_df.copy()
results_df['Predicted'] = model.predict(results_df)
results_df['Residual'] = results_df[dep_var] - results_df['Predicted']

with tabs[1]:
    st.header("Model overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared", f"{model.rsquared:.3f}")
    col2.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
    col3.metric("Observations", f"{int(model.nobs)}")
    col4.metric(f"Mean {dep_var}", f"{results_df[dep_var].mean():,.0f}")

    st.subheader("Coefficients")
    st.dataframe(coef[['term','coef','std_err','t','p_value','ci_lower','ci_upper']].applymap(lambda x: format_number(x) if isinstance(x,(int,float)) else x), use_container_width=True)

# -------------------------
# RESULTS TAB
# -------------------------
with tabs[2]:
    st.header("Interactive Results")

    fig = px.scatter(results_df, x='Predicted', y=dep_var, color_discrete_sequence=[TEAL, NAVY])
    fig.add_trace(go.Scatter(x=results_df['Predicted'], y=results_df[dep_var], mode='markers',
                             marker=dict(color=TEAL), name='Actual'))
    fig.update_layout(title="Predicted vs Actual", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)

    fig_coef = px.bar(coef[coef['term']!='Intercept'], x='coef', y='term', orientation='h',
                      error_x=coef['std_err']*2, title="Coefficients (95% CI)", color_discrete_sequence=[TEAL])
    st.plotly_chart(fig_coef, use_container_width=True)

    fig_resid = px.scatter(results_df, x='Predicted', y='Residual', color_discrete_sequence=[NAVY])
    st.plotly_chart(fig_resid, use_container_width=True)

# -------------------------
# INSIGHTS TAB
# -------------------------
with tabs[3]:
    st.header("Interpretation & Actions")
    st.markdown("Significant coefficients (p<0.05) indicate factors associated with salary differences.")
    sig_coef = coef[(coef['p_value']<0.05) & (coef['term']!='Intercept')]
    if sig_coef.empty:
        st.write("No statistically significant predictors found at p<0.05.")
    else:
        st.write(sig_coef[['term','coef','p_value']].applymap(lambda x: format_number(x) if isinstance(x,(int,float)) else x))

with st.expander("Recommended actions"):
    st.markdown("""
    1. Investigate any significant gaps in protected categories (Gender, Grouping).  
    2. Review outliers in predicted vs actual (Residuals) for underpaid/overpaid employees.  
    3. Verify data quality (Salary, JobLevel, ServiceYears, Rating).  
    4. Adjust policies/pay where justified, document rationale.
    """)

