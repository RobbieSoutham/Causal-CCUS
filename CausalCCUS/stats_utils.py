"""
causalccus.stats_utils
======================
Quick descriptive statistics and grouped summaries.
"""

from __future__ import annotations
import pandas as pd

def describe_by_group(df: pd.DataFrame,
                      group_cols: list[str],
                      target_cols: list[str]):
    """
    Return summary stats (mean, std, N) by group.

    Example
    -------
    >>> describe_by_group(df, ["Sector"], ["Emissions", "Capex"])
    """
    agg = {}
    for col in target_cols:
        agg[col] = ["mean", "std", "count"]
    return (
        df.groupby(group_cols)
          .agg(agg)
          .rename(columns={"count": "N"})
          .round(3)
    )
