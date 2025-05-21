from arch import arch_model
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def fit_garch(returns):
    """
    Ajuste un modèle GARCH(1,1) sur les rendements.
    Retourne les résidus et les volat. conditionnelles.
    """
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
    res = model.fit(disp='off')

    residuals = res.resid.dropna()
    sigma = res.conditional_volatility.dropna()

    return residuals, sigma

def standardize_residuals(residuals, sigma):
    """
    Standardise les résidus par leur volatilité conditionnelle.
    """
    standardized = residuals / sigma
    return standardized

def to_pseudo_observations(standardized_residuals):
    """
    Transforme les résidus standardisés en pseudo-observations uniformes [0, 1].
    Utilise le rang empirique divisé par (n+1) pour éviter les extrêmes 0 et 1.
    """
    n = len(standardized_residuals)
    ranks = rankdata(standardized_residuals, method="average")  # rangs
    u = ranks / (n + 1)  # transformation uniforme
    return pd.Series(u, index=standardized_residuals.index)