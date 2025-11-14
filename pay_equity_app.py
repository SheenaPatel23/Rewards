# Full corrected app.py with updated column handling, safe formula building, highlighting tabs, and V.Group branding

import base64
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# Brand Colors
# -------------------------------------------------
LIME = "#68DA6A"
TEAL = "#00A3E0"
LIGHT_GREY = "#F5F5F5"
WHITE = "#FFFFFF"

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical

def safe_formula(dep: str, indep: List[str], categorical_cols: List[str]) -> str:
    terms = []
    for c in indep:
        if " " in c:
            c_clean = f"Q('{c}')"
        else:
            c_clean = c
        if c in categorical_cols:
            terms.append(f"C({c_clean})")
        else:
            terms.append(c_clean)
    rhs = " + ".join(terms) if terms else "1"
    dep_clean = f"Q('{dep}')" if " " in dep else dep
    return f"{dep_clean} ~ {rhs}"

def run_model(formula: str, df: pd.DataFrame):
    try:
        return smf.ols(formula=formula, data=df).fit()
    except Exception as e:
        st.error(f"Model error: {e}")
        return None

def format_num(x):
    try:
        return f"{x:,.0f}"
    except:
        return x

# -------------------------------------------------
# Streamlit Layout
# -------------------------------------------------

st.set_page_config(page_title="Pay Equity Regression", layout="wide")
st.markdown(f"<h1 style='color:{LIME}'>Pay Equity Regression Dashboard</h1>", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.header("Upload Data")
upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if upload is not None:
    df = pd.read_csv(upload)
    st.sidebar.success("Data loaded successfully")
else:
    st.stop()

# Column verification
expected_cols = [
    "EmployeeID","Gender","JobLevel","Total Grouping",
    "Grouping","Grouping Category","ServiceYears",
    "TenureinRole","Rating","Salary"
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Remove Total Grouping
if "Total Grouping" in df.columns:
    df = df.drop(columns=["Total Grouping"])

# Detect types
numeric_cols, categorical_cols = detect_column_types(df)

# Sidebar Model Config
st.sidebar.header("Regression Setup")
dep_var = st.sidebar.selectbox("Dependent Variable", options=numeric_cols, index=numeric_cols.index("Salary") if "Salary" in numeric_cols else 0)

indep_default = [
    "Gender","JobLevel","Grouping","Grouping Category",
    "ServiceYears","TenureinRole","Rating"
]

indep_vars = st.sidebar.multiselect(
    "Independent Variables", options=df.columns, default=[c for c in indep_default if c in df.columns]
)

formula = safe_formula(dep_var, indep_vars, categorical_cols)
st.sidebar.code(formula, language="python")

run = st.sidebar.button("Run Regression")

# Tabs
TAB_DATA, TAB_MODEL, TAB_RESULTS, TAB_INSIGHTS = st.tabs(["📊 Data", "🧮 Model", "📈 Results", "💡 Insights"])

# -------------------------------------------------
# DATA TAB
# -------------------------------------------------

with TAB_DATA:
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)
    with st.expander("Summary Statistics"):
        st.dataframe(df.describe(include='all').T)

if not run:
    with TAB_MODEL:
        st.info("Configure model and click Run Regression.")
    st.stop()

# -------------------------------------------------
# MODEL
# -------------------------------------------------
model = run_model(formula, df)
if model is None:
    st.stop()

results = df.copy()
results["Predicted"] = model.predict(results)
results["Residual"] = results[dep_var] - results["Predicted"]

coef = model.params.reset_index()
coef.columns = ["Term", "Coefficient"]
coef["StdErr"] = model.bse.values
coef["t"] = model.tvalues.values
coef["p"] = model.pvalues.values
ci = model.conf_int()
coef["CI Lower"] = ci[0].values
coef["CI Upper"] = ci[1].values

with TAB_MODEL:
    st.subheader("Model Summary Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R-squared", f"{model.rsquared:.3f}")
    c2.metric("Adj R-squared", f"{model.rsquared_adj:.3f}")
    c3.metric("Observations", int(model.nobs))
    c4.metric(f"Mean {dep_var}", format_num(results[dep_var].mean()))

    st.subheader("Coefficients Table")
    st.dataframe(coef, use_container_width=True)

# -------------------------------------------------
# RESULTS TAB
# -------------------------------------------------
with TAB_RESULTS:
    st.subheader("Actual vs Predicted")
    fig1 = px.scatter(
        results,
        x="Predicted",
        y=dep_var,
        color_discrete_sequence=[TEAL],
        labels={dep_var:"Actual"}
    )
    fig1.add_trace(go.Scatter(
        x=results["Predicted"],
        y=results[dep_var],
        mode="markers",
        marker=dict(color=LIME),
        name="Actual"
    ))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Residuals vs Predicted")
    fig2 = px.scatter(results, x="Predicted", y="Residual", color_discrete_sequence=[LIME])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Coefficient Plot")
    coef_plot = coef[coef["Term"] != "Intercept"].copy()
    fig3 = px.bar(
        coef_plot,
        x="Coefficient",
        y="Term",
        orientation='h',
        color_discrete_sequence=[TEAL]
    )
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# INSIGHTS TAB
# -------------------------------------------------
with TAB_INSIGHTS:
    st.subheader("Key Findings & Recommended Actions")

    sig = coef[(coef["p"] < 0.05) & (coef["Term"] != "Intercept")]
    if sig.empty:
        st.info("No statistically significant predictors at p < 0.05.")
    else:
        st.write("**Significant Predictors**:")
        st.dataframe(sig, use_container_width=True)

    with st.expander("Recommended Actions"):
        st.markdown("""
        **1. Review significant drivers of pay differences.**  
        **2. Investigate any gender or grouping gaps.**  
        **3. Validate whether differences are justified by job level, service, or performance.**  
        **4. Review individuals with high positive or negative residuals.**  
        **5. Document decisions and ensure fair pay governance.**
        """)

st.markdown("---")
st.caption("Built for Rewards — Finance Data & Analytics")
