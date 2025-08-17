"""
causalccus.data_utils
=====================
Lightweight loaders coercing raw CCUS and emissions data to tidy DataFrames.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np

def _read_generic(path: str | Path, **kwargs):
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def read_ghgrp(path: str | Path):
    df = _read_generic(path)
    # harmonise column names
    return df.rename(columns=str.lower)

def read_seds(path: str | Path):
    df = _read_generic(path)
    return df.rename(columns=str.lower)


def prepare_panel_data(
    df,
    unit_vars=None,
    time_var=None,
    outcome_vars=None,
    treatment_vars=None,
    sort_rows=True
):
    """
    Prepare a clean panel DataFrame in long format with panel keys.

    Returns a sorted DataFrame with consistent types for panels.
    """
    df = df.copy()
    # Check and coerce panel keys
    if unit_vars is not None:
        df[unit_vars] = df[unit_vars].astype(str)
    if time_var is not None:
        if not np.issubdtype(df[time_var].dtype, np.number):
            df[time_var] = pd.to_numeric(df[time_var], errors='coerce')
    # Outcome/treatment columns
    for col in (outcome_vars or []) + (treatment_vars or []):
        if col not in df.columns:
            raise ValueError(f"{col} not in DataFrame")
    # Sort
    if sort_rows and unit_vars and time_var:
        df = df.sort_values(unit_vars + [time_var])
    return df

def create_event_time(
    df,
    group_vars,
    treatment_var,
    time_var,
    threshold=0
):
    """
    Adds columns: 'treated', 'first_treat_year', 'event_time'.
    - treated: binary indicator for treatment (> threshold)
    - first_treat_year: first period a unit is treated
    - event_time: period - first_treat_year
    """
     # 2. Create treatment indicators
    df = df.copy()
    df['treated'] = (df['d_log_eor_capacity'] > 0).astype(int)
    
    # 3. Create treatment year (first year unit received treatment)
    # GROUP BY COUNTRY + SECTOR for better statistical power
    groups = []
    for i, g in enumerate(group_vars):
        if i == 0:
            groups = df[g] + ('_' if len(group_vars) > 0 else '')
        else:
            groups += df[g] + ('_' if i != len(group_vars) - 1 else '')
            
    df['treatment_group'] = groups

    df['treatment_year'] = df.groupby('treatment_group')['Year'].transform(
        lambda x: x[df.loc[x.index, 'treated'] == 1].min() 
        if any(df.loc[x.index, 'treated'] == 1) else np.inf
    )
    
    # 4. Create event_time variable
    df['event_time'] = df['Year'] - df['treatment_year']
    df.loc[df['treatment_year'] == np.inf, 'event_time'] = np.nan

    print("Preprocessing completed...")
    print(f"Treated units: {df['treated'].sum()}")
    print(f"Control units: {(df['treated'] == 0).sum()}")
    print(f"Treatment groups: {df['treatment_group'].nunique()}")
    print(f"Event time range: {df['event_time'].min():.0f} to {df['event_time'].max():.0f}")

    return df

def subset_by_control_counts(
    df,
    treatment_var='d_log_eor_capacity',
    time_var='Year',
    group_vars=('Country', 'Sector'),
    min_clusters=5,
    min_obs=30,
    min_controls=10
):
    """
    Only keeps (unit, year) cells with enough not‐yet‐treated controls.
    Returns dict of:
      - filtered_df
      - group_stats
      - cell_stats
      - keep_cells
    """
    df = df.copy()
    df['treated'] = (df[treatment_var] > 0).astype(int)
    cohort = (
        df[df['treated'] == 1].groupby(list(group_vars))[time_var].min().reset_index(name='cohort')
    )
    all_grps = df[list(group_vars)].drop_duplicates()
    cohort = all_grps.merge(cohort, on=list(group_vars), how='left')
    cohort['cohort'] = cohort['cohort'].fillna(np.inf)
    group_stats = (
        df.groupby(list(group_vars)).agg(
            n_obs=('treated','size'), n_treated=('treated','sum')
        ).reset_index()
    )
    valid_groups = group_stats[
        (group_stats['n_treated'] >= min_clusters) &
        (group_stats['n_obs'] >= min_obs)
    ][group_vars]
    df2 = df.merge(valid_groups, on=list(group_vars), how='inner')
    cohort2 = cohort.merge(valid_groups, on=list(group_vars), how='inner')
    years = sorted(df2[time_var].unique())
    cells = []
    for _, row in cohort2.iterrows():
        grp = tuple(row[var] for var in group_vars)
        g_cohort = row['cohort']
        for yr in years:
            n_controls = (cohort2['cohort'] > yr).sum()
            cells.append({**{var: grp[i] for i, var in enumerate(group_vars)}, time_var: yr, 'n_controls': n_controls})
    cell_stats = pd.DataFrame(cells)
    keep_cells = cell_stats[cell_stats['n_controls'] >= min_controls]
    key_cols = list(group_vars) + [time_var]
    df_keep = df2.merge(keep_cells[key_cols], on=key_cols, how='inner')
    return {
        'filtered_df': df_keep,
        'group_stats': group_stats,
        'cell_stats': cell_stats,
        'keep_cells': keep_cells
    }

def get_event_df(
    df_orig,
    group_vars=['Country', 'Sector'],
    treatment_var='d_log_eor_capacity'
):
    df = df_orig.copy()
    df['treated'] = (df[treatment_var] > 0).astype(int)
    # Create treatment_groups as concatenation of group_vars
    groups = df[group_vars[0]].astype(str)
    for g in group_vars[1:]:
        groups += "_" + df[g].astype(str)
    df['treatment_group'] = groups
    # First treatment year per group
    df['treatment_year'] = df.groupby('treatment_group')['Year'].transform(
        lambda x: x[df.loc[x.index, 'treated'] == 1].min() 
        if any(df.loc[x.index, 'treated'] == 1) else np.inf
    )
    # Event time relative to first treatment
    df['event_time'] = df['Year'] - df['treatment_year']
    df.loc[df['treatment_year'] == np.inf, 'event_time'] = np.nan
    return df
