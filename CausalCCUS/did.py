# ==============================================================
# 0. Imports
# ==============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from typing import Literal, Optional
import matplotlib.pyplot as plt
import seaborn as sns 

# ==============================================================
# 1. Helper – add post & interaction
# ==============================================================

def build_did_design_matrix(
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treat: str,
) -> pd.DataFrame:
    """
    Adds `post` and `did_interaction` columns.
    `post_it = 1` iff year >= *first* treated year *for that unit*.
    """
    g = df.copy()
    # First treatment time per unit
    first_treat = (
        g.loc[g[treat] != 0, [unit, time]]
        .groupby(unit)[time]
        .min()
        .rename("first_treat_year")
    )
    g = g.merge(first_treat, on=unit, how="left")
    g["post"] = (g[time] >= g["first_treat_year"]).astype(int).fillna(0)
    g["did_interaction"] = g["post"] * g[treat]
    return g.drop(columns="first_treat_year")

# ==============================================================
# 2. ATTᴼ – remains unchanged
# ==============================================================

def att_overall(
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treat: str,
    method: Literal["ols", "panelols"] = "panelols",
    cluster: Optional[str] = None,
):
    design = build_did_design_matrix(df, outcome, unit, time, treat)

    if method == "ols":
        fe_terms = f"C({unit}) + C({time})"
        fml = f"{outcome} ~ post + {treat} + did_interaction + {fe_terms}"
        return smf.ols(fml, design).fit(
            cov_type="cluster",
            cov_kwds={"groups": design[cluster or unit]},
        )

    if method == "panelols":
        panel = design.set_index([unit, time])
        exog = panel[["post", treat, "did_interaction"]]
        res = PanelOLS(
            panel[outcome],
            exog,
            entity_effects=True,
            time_effects=True,
        ).fit(cov_type="clustered", cluster_entity=True)
        return res

    raise ValueError("method must be 'ols' or 'panelols'")

# ==============================================================
# 3. Event-Study (PTA) – continuous vs. binary flag
# ==============================================================
def event_study_for_pta(
    df: pd.DataFrame,
    outcome_var: str = "d_log_emissions",
    treatment_var: str = "d_log_eor_capacity",
    event_time_var: str = "event_time",
    covariates: Optional[list[str]] = None,
    cluster_var: str = "treatment_group",
    pre_window: int = 5,
    post_window: int = 5
) -> dict:
    """
    Event study for PTA testing (Callaway et al., 2024).
    Omits only t = -1 as the reference period.
    """
    data = df.copy()
    # 1. Binarize treatment
    data["treated"] = (data[treatment_var] > 0).astype(int)
    # 2. Filter to event window
    data = data.loc[
        data[event_time_var].between(-pre_window, post_window) &
        data[event_time_var].notna()
    ].copy()
    if data.empty:
        return {"model": None, "interaction_terms": [], "p_value": np.nan, "error": "No data"}

    # 3. Build interactions, omitting only t = -1
    interaction_terms = []
    for t in sorted(data[event_time_var].unique()):
        if t == -1:
            continue  # drop only t = -1
        name = f"event_{int(t)}" if t >= 0 else f"event_m{abs(int(t))}"
        data[name] = (data[event_time_var] == t).astype(int)
        inter = f"{name}_treated"
        data[inter] = data[name] * data["treated"]
        interaction_terms.append(inter)

    # 4. Build formula
    rhs = interaction_terms + (covariates or [])
    formula = f"{outcome_var} ~ " + " + ".join(rhs)
    # 5. Fit model with cluster-robust SEs
    fit_kwargs = {}
    if cluster_var in data:
        fit_kwargs = {"cov_type": "cluster", "cov_kwds": {"groups": data[cluster_var]}}
    model = smf.ols(formula, data=data).fit(**fit_kwargs)

    # 6. PTA test on pre-treatment interactions (t < 0, excluding -1)
    pre_terms = [term for term in interaction_terms if term.startswith("event_m")]
    if pre_terms:
        f_test = model.f_test([f"{pt} = 0" for pt in pre_terms])
        pval = float(f_test.pvalue)
        method = "F-test"
    else:
        pval = np.nan
        method = "No pre-treatment terms"

    return {
        "model": model,
        "interaction_terms": interaction_terms,
        "pre_terms": pre_terms,
        "p_value": pval,
        "test_method": method,
        "n_pre_periods": len(pre_terms),
        "n_post_periods": len(interaction_terms) - len(pre_terms),
    }


def att_o_estimator(
    df: pd.DataFrame,
    outcome_var: str = "d_log_emissions",
    treatment_var: str = "d_log_eor_capacity",
    covariates: Optional[list[str]] = None,
    cluster_var: str = "treatment_group"
) -> dict:
    """
    Binarized ATT^o estimator (Callaway et al., 2024).
    """
    data = df.copy()
    data["treated"] = (data[treatment_var] > 0).astype(int)
    covs = covariates or []
    formula = f"{outcome_var} ~ treated" + (" + " + " + ".join(covs) if covs else "")
    fit_kwargs = {}
    if cluster_var in data:
        fit_kwargs = {"cov_type": "cluster", "cov_kwds": {"groups": data[cluster_var]}}
    model = smf.ols(formula, data=data).fit(**fit_kwargs)
    return {
        "model": model,
        "att_o": float(model.params["treated"]),
        "std_error": float(model.bse["treated"]),
        "conf_int": model.conf_int().loc["treated"].tolist(),
        "p_value": float(model.pvalues["treated"]),
    }

# ==============================================================
# 6. Honest DiD Bounds – unchanged
# ==============================================================

def honest_did_bounds(
    pta_results: dict,
    att_estimate: float,
    std_error: float,
    M_values = (0.5, 1, 1.5, 2)
) -> dict:
    pre_coefs = [pta_results["model"].params[p] for p in pta_results["pre_terms"]]
    v = max(abs(c) for c in pre_coefs) if pre_coefs else 0
    out = {}
    for M in M_values:
        shift = M * v
        lo = att_estimate - 1.96*std_error - shift
        hi = att_estimate + 1.96*std_error + shift
        out[f"M={M}"] = (lo, hi)
    return out
