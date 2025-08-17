"""
causalccus.vif
==============
Tools for detecting multicollinearity in cross-section or panel regressors.
"""

from __future__ import annotations
import pandas as pd
from CausalCCUS.constants import FIGURE_DIR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def get_VIF_values(
        df: pd.DataFrame,
        predictors: list,
        save_file: str = None
        ) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for a set of predictors in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    predictors : list of str
        List of column names in df to include in VIF computation.
    
    Returns:
    --------
    vif_df : pandas.DataFrame
        DataFrame with columns ['feature', 'VIF'] listing each predictor 
        and its corresponding VIF.
    """
    # Subset the DataFrame to the predictors
    X = df[predictors].fillna(0).copy()
    
    # Add constant term for intercept
    X['const'] = 1
    
    # Compute VIF for each predictor
    vif_data = []
    for i, feature in enumerate(X.columns):
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append({'feature': feature, 'VIF': vif_val})
    
    vif_df = pd.DataFrame(vif_data).sort_values(
        by='VIF', ascending=False
        ).reset_index(drop=True)
    
    if save_file:
        vif_df.to_csv(FIGURE_DIR + save_file)
    return vif_df