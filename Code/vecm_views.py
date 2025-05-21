from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
import pandas as pd
import numpy as np

def fit_vecm(price_data, lags=1, coint_rank=None):
    """
    Estime un modèle VECM (Johansen) sur les prix.

    Inputs:
        price_data : DataFrame de prix (colonnes = actifs, lignes = dates)
        lags : nombre de retards à utiliser dans le VECM
        coint_rank : nombre de relations de co-intégration (défaut : déterminé automatiquement)

    Output:
        forecast : prédiction des prix à t+1 (série pandas)
    """
    # Supprimer les lignes NaN
    price_data = price_data.dropna()

    # Test de co-intégration de Johansen
    johansen_result = coint_johansen(price_data, det_order=0, k_ar_diff=lags)

    if coint_rank is None:
        # Choisir le rang de co-intégration selon la statistique de trace
        trace_stats = johansen_result.lr1
        crit_vals = johansen_result.cvt[:, 1]  # 5% critical value
        coint_rank = np.sum(trace_stats > crit_vals)

    # VECM fit
    model = VECM(price_data, k_ar_diff=lags, coint_rank=coint_rank, deterministic="n")
    vecm_res = model.fit()

    # Prévision à t+1 (en niveau)
    forecast = vecm_res.predict(steps=1)
    forecast_df = pd.Series(forecast[0], index=price_data.columns, name="Forecast")

    return forecast_df


def generate_views(current_prices: pd.Series, forecast_prices: pd.Series) -> tuple:
    """
    Génére la matrice P (identité) et le vecteur q (vues directionnelles) à partir :
    - des prix actuels,
    - des prix prévus à t+1 (VECM).

    Output :
        P : matrice identité de taille (n, n)
        q : rendement directionnel prévisionnel = (p_forecast / p_current) - 1
    """
    tickers = current_prices.index.tolist()
    n_assets = len(tickers)

    # Matrice identité : chaque vue porte sur un seul actif
    P = np.eye(n_assets)

    # Vues directionnelles en rendement log ou simple
    q = (forecast_prices / current_prices) - 1
    q = q.values.reshape(-1, 1)

    return P, q
