"""
causalccus.plotting
===================
High-level plotting utilities for CCUS analytics.
"""

from __future__ import annotations
import re
import pandas as pd
from CausalCCUS.constants import FIGURE_DIR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict

class FigFinaliser:
    """
    Decorator to finalize a matplotlib figure: shows it and optionally saves to a file.
    Wrapped functions may accept a `save_path: str` kwarg.
    """
    def __init__(self):
        pass

    def __call__(self, func):
        def wrapper(*args, save_path: str = None, **kwargs):
            # Call the plotting function, passing along save_path if desired
            result = func(
                *args, **{**kwargs, **(
                    {} if save_path is None else {"save_path": save_path}
                    )}
            )
            if save_path:
                plt.savefig(FIGURE_DIR + save_path, bbox_inches="tight", dpi=200)
            plt.show()
            return result
        return wrapper

def _clean_policy_dates(policy_dates: Optional[Dict[str, float | int]]):
    clean = {}
    if policy_dates:
        for event, yr in policy_dates.items():
            yr0 = np.atleast_1d(yr).astype(float)[0]
            if np.isfinite(yr0):
                clean[event] = yr0
    return clean

@FigFinaliser()
def plot_by_sector_with_policies(df: pd.DataFrame,
                                 value_col: str = "Value",
                                 sector_col: str = "Sector",
                                 year_col: str = "Year",
                                 policy_dates: Optional[Dict[str, float | int]] = None,
                                 ylabel: str = "Value",
                                 title: str = "Trends by Sector",
                                 palette: Optional[str] = "tab10"):
    policy_dates = _clean_policy_dates(policy_dates)

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df,
        x=year_col,
        y=value_col,
        hue=sector_col,
        ci=None,
        alpha=0.7,
        palette=palette,
    )

    ymin, ymax = ax.get_ylim()
    for event, yr in policy_dates.items():
        ax.vlines(yr, ymin, ymax, color="red", linestyle="--", alpha=0.7)
        ax.text(yr, ymax * 0.95, event, rotation=90, va="top", ha="right",
                fontsize=11, color="black")

    
    ax.set_xlabel(year_col, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    #ax.set_title(title, fontsize=16, weight="bold")
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(title=sector_col, bbox_to_anchor=(1.05, 1), loc="upper left")

    return ax

@FigFinaliser()
def plot_corr_matrix(
        df: pd.DataFrame,
        cols: list[str],
        method: str = "pearson",
        mask_triangle: bool = True,
        cmap: str = "coolwarm",
        title: str = "Correlation Matrix",
        save_path: str = None
    ):
    """
    Visualise pairwise correlations.

    Parameters
    ----------
    cols : list[str]
        Columns to include.
    method : {'pearson', 'spearman', 'kendall'}
    mask_triangle : bool
        If True masks upper triangle for clarity.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    corr = df[cols].corr().round(1)
    
    sns.heatmap(
        data=corr,
        cmap='coolwarm',
        annot=True,
        ax=ax,
        annot_kws={"size": 16},
    )

    plt.tick_params(axis='both', which='major', labelsize=16)  # Font size for tick labels

    corr.to_csv('corr.csv')

@FigFinaliser()
def plot_event_study(
    results: dict,
    title_suffix: str = "",
    save_path: str = None
    ):
    """
    Plot event-study coefficients with 95% CIs.
    """

    model = results.get("model")
    terms = results.get("interaction_terms", [])
    if model is None or not terms:
        print("No results to plot.")
        return None

    rows = []
    for term in terms:
        if term not in model.params:
            continue
        m = re.fullmatch(r"(?:event_m(\d+)|event_(\d+))_treated", term)
        if not m:
            continue
        neg, pos = m.groups()
        t = -int(neg) if neg else int(pos)
        coef = float(model.params[term])
        se = float(model.bse[term])
        rows.append((t, coef, coef - 1.96*se, coef + 1.96*se))

    if not rows:
        print("No valid coefficients.")
        return None

    dfp = pd.DataFrame(rows, columns=["t","coef","lo","hi"]).sort_values("t")
    fig, ax = plt.subplots(figsize=(10,6))
    pre = dfp[dfp.t < 0]
    post = dfp[dfp.t >= 0]
    if not pre.empty:
        ax.errorbar(pre.t, pre.coef, 
                    yerr=[pre.coef - pre.lo, pre.hi - pre.coef],
                    fmt="o-", color="navy", ecolor="skyblue", capsize=4, label="Pre")
    if not post.empty:
        ax.errorbar(post.t, post.coef, 
                    yerr=[post.coef - post.lo, post.hi - post.coef],
                    fmt="o-", color="darkorange", ecolor="bisque", capsize=4, label="Post")
    
    #ax.plot(dfp.t, dfp.coef, color="gray", alpha=0.5)
    
    ax.axhline(0, color="grey", linestyle="--")
    ax.axvline(-1, color="red", linestyle=":")
    
    ax.set_xlabel("Years Relative to Treatment (t=-1)", fontsize=24)
    ax.set_ylabel("Treatment Effect", fontsize=24)
    ax.set_xticks(dfp.t.tolist())
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    ax.legend()
    
    #plt.title(title_suffix or "Event-Study Plot")

@FigFinaliser()
def plot_raw_trends(
    df: pd.DataFrame,
    outcome_var: str = "d_log_emissions",
    time_var: str = "event_time",
    group_var: str = "treated",
    event_window: tuple[int, int] = (-5, 10),
    save_path: str = None
):
    """
    Overlay mean outcome for treated vs. control over event_time with error bands.
    """
    plot_df = df[
        df[time_var].between(event_window[0], event_window[1])
    ].copy()
    
    # Calculate mean and standard error for each group and time
    summary = (
        plot_df
        .groupby([time_var, group_var])[outcome_var]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    
    # Calculate standard error
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['lower_ci'] = summary['mean'] - summary['se']
    summary['upper_ci'] = summary['mean'] + summary['se']
    
    plt.figure(figsize=(10, 6))
    
    # Create the plot with error bands
    for group in summary[group_var].unique():
        group_data = summary[summary[group_var] == group]
        
        # Plot the line
        plt.plot(
            group_data[time_var], 
            group_data['mean'], 
            marker="o", 
            label=f"{group_var}={group}",
            linewidth=2
        )
        
        # Add error bands (shaded region)
        plt.fill_between(
            group_data[time_var], 
            group_data['lower_ci'], 
            group_data['upper_ci'], 
            alpha=0.3
        )
    
    plt.ylabel(outcome_var, fontsize=24)
    plt.xlabel('Years Relative to Treatment (t=-1)', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.axvline(-1, color="red", linestyle="--", alpha=0.7)
    plt.axhline(0, color="grey", linestyle="--", alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()

