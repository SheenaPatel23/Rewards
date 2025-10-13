"""
Streamlit Pay Equity Regression App
==================================
Single-file Streamlit app that lets Rewards / HR analysts:
 - Upload their employee compensation CSV (or use sample data)
 - Select dependent variable (e.g. Salary) and independent variables
 - Run an OLS regression (statsmodels) and view coefficients, p-values, R-squared
 - See predicted vs actual scatter, residuals table, coefficient bar chart
 - Download employee-level predictions and coefficients

How to run
----------
1. Save this file as `streamlit_pay_equity_app.py`.
2. Create virtualenv and install dependencies:
   pip install streamlit pandas numpy matplotlib statsmodels scikit-learn
3. Run:
   streamlit run streamlit_pay_equity_app.py
"""

import io
import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

# -------------------------
# Helper functions
# -------------------------

def load_sample_data() -> pd.DataFrame:
    """Return a small realistic sample dataset for pay equity analysis."""
    d = {
        'EmployeeID': [f'E{str(i).zfill(4)}' for i in range(1,101)],
        'Gender': np.random.choice(['Male','Female'], size=100, p=[0.55,0.45]),
        'JobLevel': np.random.choice([2,3,4,5,6], size=100, p=[0.1,0.25,0.35,0.2,0.1]),
        'Grouping': np.random.choice(['Ops','Sales','Finance','HR','IT'], size=100),
        'ServiceYears': np.round(np.random.exponential(scale=5, size=100)).astype(int),
        'TenureinRole': np.round(np.random.exponential(scale=2, size=100)).astype(int),
        'Rating': np.random.choice([1,2,3,4,5], size=100, p=[0.05,0.15,0.45,0.25,0.1]),
        'Salary': 0  # placeholder, will construct below
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
    # Introduce a small unexplained gender gap for demo
    df.loc[df['Gender']=='Female', 'Salary'] *= 0.97
    df['Salary'] = (df['Salary'] + np.random.normal(scale=3000, size=len(df))).round(0)

    return df

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) for UI suggestions."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical

def build_formula(dep: str, indep: List[str]) -> str:
    """Construct a formula string for statsmodels."""
    rhs = ' + '.join(indep) if indep else '1'
    return f"{dep} ~ {rhs}"

def run_ols_formula(formula: str, df: pd.DataFrame):
    """Fit OLS using statsmodels and return the results object."""
    model = smf.ols(formula=formula, data=df).fit()
    return model

def plot_coefficients(coeff_df: pd.DataFrame):
    """Bar chart of coefficients with error bars (2*std err)."""
    fig, ax = plt.subplots(figsize=(8,4))
    coeff_df = coeff_df.copy()
    coeff_df = coeff_df.sort_values('coef')
    ax.barh(coeff_df['term'], coeff_df['coef'])
    ax.set_xlabel('Coefficient')
    ax.set_ylabel('Variable')
    ax.set_title('Regression Coefficients')
    plt.tight_layout()
    return fig

def plot_predicted_vs_actual(df: pd.DataFrame, dep: str, pred_col: str = 'Predicted'):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df[pred_col], df[dep], alpha=0.7)
    mx = np.linspace(min(df[pred_col].min(), df[dep].min()), max(df[pred_col].max(), df[dep].max()), 100)
    ax.plot(mx, mx, linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Predicted vs Actual')
    plt.tight_layout()
    return fig

def to_download_link(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"data:file/csv;base64,{b64}"
    return href

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Rewards - Pay Equity Regression", layout='wide')
st.title("Rewards — Pay Equity Regression Dashboard")
st.markdown(
    "Upload a CSV with employee-level data or use the included sample dataset.\n"
    "Select dependent variable (usually Salary) and explanatory variables."
)

# Sidebar controls
st.sidebar.header('Data')
upload = st.sidebar.file_uploader("Upload CSV file (employee-level)", type=['csv'])
use_sample = st.sidebar.checkbox('Use sample dataset', value=True if upload is None else False)

if upload is not None:
    try:
        raw_df = pd.read_csv(upload)
        st.sidebar.success('File loaded successfully')
    except Exception as e:
        st.sidebar.error(f'Error reading CSV: {e}')
        st.stop()
elif use_sample:
    raw_df = load_sample_data()
else:
    st.warning('Please upload a CSV or choose the sample dataset')
    st.stop()

# Show a slice of the data
st.subheader('Data preview')
st.dataframe(raw_df.head(200))

numeric_cols, categorical_cols = detect_column_types(raw_df)

with st.sidebar.form('model_options'):
    st.header('Model configuration')
    dep_var = st.selectbox('Dependent variable (target)', options=numeric_cols, index= numeric_cols.index('Salary') if 'Salary' in numeric_cols else 0)

    st.markdown('**Choose independent variables (features)**')
    default_indep = [c for c in ['JobLevel','Gender','Grouping','ServiceYears','TenureinRole','Rating'] if c in raw_df.columns]
    indep_vars = st.multiselect('Independent variables', options=[*numeric_cols, *categorical_cols], default=default_indep)

    st.markdown('''If a variable is categorical (e.g. Gender, Grouping), the app will automatically
        treat it as categorical by wrapping it using `C(variable)` in the regression formula.
        You can also manually edit the formula below before running.
    ''')

    # Build suggested formula automatically (wrap non-numeric indep in C())
    indep_for_formula = []
    for v in indep_vars:
        if v in categorical_cols:
            indep_for_formula.append(f"C({v})")
        else:
            indep_for_formula.append(v)

    suggested_formula = build_formula(dep_var, indep_for_formula)
    formula = st.text_input('Regression formula (editable)', value=suggested_formula, help='You can edit the formula, e.g. add interactions like C(Grouping):JobLevel')

    test_size = st.slider('Test set size (%)', min_value=0, max_value=50, value=20, step=5)
    btn_run = st.form_submit_button('Run regression')

if not btn_run:
    st.info('Configure the model on the left and click `Run regression`')
    st.stop()

# Run regression
st.subheader('Model results')
try:
    # Train/test split
    if test_size > 0:
        train_df, test_df = train_test_split(raw_df, test_size=test_size/100, random_state=42)
    else:
        train_df = raw_df.copy()
        test_df = pd.DataFrame(columns=raw_df.columns)

    model = run_ols_formula(formula, train_df)

    st.markdown('**Summary**')
    st.write(f"R-squared: {model.rsquared:.4f} | Adj. R-squared: {model.rsquared_adj:.4f}")
    st.write(f"N (observations): {int(model.nobs)}")

    # Coefficients table
    coef = model.params.reset_index()
    coef.columns = ['term','coef']
    coef['std_err'] = model.bse.values
    coef['t'] = model.tvalues.values
    coef['p_value'] = model.pvalues.values
    coef['ci_lower'] = model.conf_int().loc[:,0].values
    coef['ci_upper'] = model.conf_int().loc[:,1].values

    st.markdown('**Coefficients**')
    st.dataframe(coef)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_coefficients(coef[coef['term'] != 'Intercept'])
        st.pyplot(fig1)
    with col2:
        # Prepare predictions
        train_df['Predicted'] = model.predict(train_df)
        fig2 = plot_predicted_vs_actual(train_df, dep=dep_var, pred_col='Predicted')
        st.pyplot(fig2)

    # Residuals
    st.markdown('**Residuals (train)**')
    train_df['Residual'] = train_df[dep_var] - train_df['Predicted']
    residuals_sorted = train_df[['EmployeeID'] + [dep_var, 'Predicted', 'Residual']] if 'EmployeeID' in train_df.columns else train_df[[dep_var, 'Predicted', 'Residual']]
    st.dataframe(residuals_sorted.sort_values('Residual'))

    # Test set evaluation if available
    if not test_df.empty:
        st.markdown('**Test set evaluation**')
        test_df['Predicted'] = model.predict(test_df)
        test_df['Residual'] = test_df[dep_var] - test_df['Predicted']
        mse = ((test_df['Residual'])**2).mean()
        st.write(f"Test MSE: {mse:.2f}")
        fig_test = plot_predicted_vs_actual(test_df, dep=dep_var, pred_col='Predicted')
        st.pyplot(fig_test)

    # Downloads
    st.markdown('**Download results**')
    coeff_href = to_download_link(coef, 'regression_coefficients.csv')
    st.markdown(f"[Download coefficients CSV]({coeff_href})")

    out_df = train_df.copy()
    if not test_df.empty:
        out_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    results_href = to_download_link(out_df, 'employee_regression_results.csv')
    st.markdown(f"[Download employee-level predictions CSV]({results_href})")

    # Quick interpretation guidance
    st.markdown('''
    **Quick interpretation guidance**
    - Focus on variables with p < 0.05 for robust signals.
    - The coefficient for `C(Gender)[T.Female]` (or similar) shows the *average* difference
      associated with the female category after controlling for other factors.
    - Residuals highlight individual employees whose pay deviates from model expectations —
      investigate these as potential outliers or data quality issues.
    ''')

except Exception as e:
    st.error(f'Error running regression: {e}')
    st.stop()

# Footer
st.markdown('---')
st.caption('Built for the Rewards team — Finance Data & Analytics')
