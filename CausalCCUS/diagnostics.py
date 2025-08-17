"""
causalccus.diagnostic
=====================
Validity checks: Parallel Trends Assumption (PTA) and covariate balance.
"""

from __future__ import annotations
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import numpy as np


def pretrend_test(
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treat: str,
    event_time: str = "event_time",
    pre_window: int = 5
) -> dict:
    """
    Wald F-test that pre-treatment event-time × treatment interactions are jointly zero.
    Assumes `df` already contains:
      - a binary `treated` column,
      - an integer `event_time` column from create_event_time.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with event_time and treated.
    outcome : str
        Outcome variable name.
    unit : str
        Panel unit identifier (unused here but kept for signature consistency).
    time : str
        Original time variable (unused here).
    treat : str
        Name of the binary treatment indicator column ("treated").
    event_time : str
        Name of the integer event_time column.
    pre_window : int
        Number of leads (negative event_time) to include, e.g. 5 tests t = -5,…,-1.

    Returns
    -------
    dict
        {
          "F": float,           # Wald F‐statistic
          "p": float,           # p‐value
          "df_denom": int,      # denominator degrees of freedom
          "df_num": int         # numerator degrees of freedom
        }
    """
    # restrict to pre‐treatment window
    df_pre = df[(df[event_time] >= -pre_window) & (df[event_time] < 0)].copy()
    if df_pre.empty:
        return {"F": float('nan'), "p": float('nan'), "df_denom": 0, "df_num": 0}

    # build interaction dummies for each lead
    leads = sorted(df_pre[event_time].unique())
    dummies = []
    for t in leads:
        col = f"int_{t}"
        df_pre[col] = ((df_pre[event_time] == t) & (df_pre[treat] == 1)).astype(int)
        dummies.append(col)

    # add intercept
    exog = sm.add_constant(df_pre[dummies])
    endog = df_pre[outcome]

    # fit OLS
    model = sm.OLS(endog, exog).fit()

    # prepare hypothesis R·β = 0 (omit intercept)
    R = []
    for i in range(len(dummies)):
        row = [0] * (len(dummies) + 1)
        row[i + 1] = 1
        R.append(row)

    f_test = model.f_test(R)
    return {
        "F": float(f_test.fvalue),
        "p": float(f_test.pvalue),
        "df_denom": int(f_test.df_denom),
        "df_num": int(f_test.df_num),
    }

def did_balance_report(
    df, 
    cluster_col=None, 
    time_col='Year', 
    treatment_var='d_log_eor_capacity',
    outcome_col='d_log_emissions',
    alpha=0.05,
    power=0.8
):
    """
    Checks panel balance, treatment assignment (continuous→binary),
    and computes Minimum Detectable Effect (MDE) for DiD.
    """
    df = df.copy()
    df['treated'] = (df[treatment_var] > 0).astype(int)
    N = len(df)
    years = sorted(df[time_col].unique())
    print(f"Total observations: {N}")
    print(f"Time span: {years[0]} – {years[-1]} ({len(years)} periods)")
    prop_treated = df['treated'].mean()
    print(f"Overall treated proportion: {prop_treated:.2%}")
    print("\nTreatment share by year:")
    print(df.groupby(time_col)['treated'].mean().to_string())
    if cluster_col and cluster_col in df.columns:
        clusters = df[cluster_col].nunique()
        ever = df.groupby(cluster_col)['treated'].max()
        print(f"\nNumber of clusters: {clusters}")
        print(f"Ever-treated clusters: {(ever>0).sum()}, Never-treated: {(ever==0).sum()}")
        print("\nObs per cluster (largest 5):")
        print(df.groupby(cluster_col).size().nlargest(5).to_string())
        print("\nObs per cluster (smallest 5):")
        print(df.groupby(cluster_col).size().nsmallest(5).to_string())
        both = df.groupby([cluster_col, time_col])['treated'].nunique() > 1
        print(f"\nCluster–year cells with both treated & control: {both.sum()}")
    else:
        print("\nNo valid cluster_col provided; skipping cluster diagnostics.")
    # MDE calculation (see notebook for full context)
    z_alpha = 1.96
    z_power = 0.84
    n_periods = len(years)
    n_clusters = df[cluster_col].nunique() if (cluster_col and cluster_col in df.columns) else N
    var_y = df[outcome_col].var()
    eff_n = n_clusters * n_periods * prop_treated * (1 - prop_treated)
    if eff_n <= 0 or var_y == 0 or np.isnan(var_y):
        print("\nCannot compute MDE: insufficient variation or zero outcome variance.")
        return
    mde = (z_alpha + z_power) * np.sqrt(4 * var_y / eff_n)
    print("\n---")
    print(f"MDE (80% power, 5% α): {mde:.3f} SDs of `{outcome_col}`")
    print(f"(clusters used: {n_clusters}, periods: {n_periods}, treated prop.: {prop_treated:.2%})")
