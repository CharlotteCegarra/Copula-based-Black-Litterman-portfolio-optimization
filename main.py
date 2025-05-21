import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from Code.data_loader import load_data, compute_log_returns
from Code.garch_models import fit_garch, standardize_residuals, to_pseudo_observations
from Code.copula_models import fit_vine_copula, simulate_joint_returns, fit_copula_clayton, fit_copula_student
from Code.vecm_views import fit_vecm, generate_views
from Code.black_litterman import compute_equilibrium_return, compute_posterior, generate_posterior_returns
from Code.optimization import max_sharpe_portfolio, min_cvar_portfolio, max_starr_portfolio

import warnings
from arch.__future__ import reindexing  # optionnelle selon ta version
from arch.univariate.base import DataScaleWarning
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=DataScaleWarning)
warnings.filterwarnings("ignore", category=ValueWarning)

def main():
    folder = "Data/"
    files = ["BNP.csv", "Airbus.csv", "Deutsche.csv", "Enel.csv", "LVMH.csv", "Sanofi.csv"]

    pseudo_obs_all = pd.DataFrame()
    all_log_returns = {}
    all_sigmas = {}

    # 🔁 Boucle sur chaque actif pour calculer les pseudo-observations
    for file in files:
        ticker = file.replace(".csv", "")

        # 1. Chargement des prix
        df = load_data(os.path.join(folder, file))
        log_returns = compute_log_returns(df["Close"])
        all_log_returns[ticker] = log_returns
    print("\n Étape 1 complétée : chargement des prix et calcul des rendements logarithmiques pour les 6 actions.")

    for file in files:
        ticker = file.replace(".csv", "")
        
        # 2. Modélisation GARCH
        log_returns = all_log_returns[ticker]
        residuals, sigma = fit_garch(log_returns)
        standardized = standardize_residuals(residuals, sigma)
        all_sigmas[ticker] = sigma
        all_log_returns[ticker] = log_returns  # (réassigné au cas où modifié)
    print("\n Étape 2 complétée : estimation des modèles GARCH(1,1) et extraction des résidus standardisés pour chaque série.")

    for file in files:
        ticker = file.replace(".csv", "")
        
        # 3. Pseudo-observations (uniformes)
        standardized = standardize_residuals(
            all_log_returns[ticker] - all_log_returns[ticker].mean(),  # ou utiliser les résidus directement si stockés
            all_sigmas[ticker]
        )
        pseudo = to_pseudo_observations(standardized)
        pseudo_obs_all[ticker] = pseudo
    print("\n Étape 3 complétée : transformation des résidus standardisés en pseudo-observations uniformes (copule-ready).")

    print("\n Aperçu des pseudo-observations (top 5 lignes) :")
    print(pseudo_obs_all.head())


    # 🔗 Ajustement de la Vine Copula multivariée
    print("\n Étape 4 : ajustement d'une copule Vine multivariée sur les 6 actifs.")
    vine_copula = fit_vine_copula(pseudo_obs_all)
    print("Copule Vine ajustée avec succès.")

    print("\n Simulation de 1000 rendements multivariés à partir de la copule et des modèles GARCH.")
    sim_returns = simulate_joint_returns(vine_copula, all_sigmas, all_log_returns, n_sim=1000)
    print("\n Aperçu des rendements simulés (top 5 lignes) :")
    print(sim_returns.head())

    # Chargement des prix actuels 
    price_data = pd.read_csv("Data/prices_aligned.csv", index_col=0, parse_dates=True)
    price_data = price_data[["BNP", "Airbus", "Deutsche", "Enel", "LVMH", "Sanofi"]]  # matching tickers

    print("\n Étape 5 : prévision des prix futurs avec modèle VECM (Vector Error Correction Model).")
    forecast = fit_vecm(price_data)
    print("\n Prévision VECM (t+1) effectuée :")
    print(forecast)

    current_prices = price_data.iloc[-1]
    forecast_prices = forecast

    P, q = generate_views(current_prices, forecast_prices)

    print("\n Génération des vues directionnelles pour Black–Litterman à partir des prévisions de prix.")
    print("\n Matrice P (structure des vues) :")
    print(P)
    print("\n Vecteur q (attentes sur les actifs) :")
    print(q)
 
    print("\n Étape 6 : calcul des rendements espérés et simulation avec modèle Black–Litterman.")
    # Matrice de covariance estimée sur les rendements
    returns_matrix = pd.DataFrame(all_log_returns).dropna()
    cov_matrix = returns_matrix.cov().values

    # Pondérations de marché (ici : équipondéré)
    market_weights = np.ones(len(returns_matrix.columns)) / len(returns_matrix.columns)

    # ✅ Étape 5.1 : rendements d’équilibre
    pi = compute_equilibrium_return(cov_matrix, market_weights, delta=2.5)

    # ✅ Étape 5.2 : fusion avec les vues (Black–Litterman)
    mu_post, cov_post = compute_posterior(pi, cov_matrix, P, q, tau=0.05)
 
    # ✅ Étape 5.3 : génération des rendements simulés
    simulated_bl_returns = generate_posterior_returns(mu_post, cov_post, n_sim=1000)

    # Résultats
    print("\n Moyennes postérieures des rendements (BL):")
    print(mu_post)

    print("\n Simulation Black–Litterman – Top 5 observations :")
    df_bl = pd.DataFrame(simulated_bl_returns, columns=returns_matrix.columns)
    print(df_bl.head())


    print("\n Étape 7 : optimisation de portefeuille selon plusieurs critères de risque-rendement.")

    # Sharpe
    w_sharpe, ret_sharpe, risk_sharpe = max_sharpe_portfolio(df_bl.values)
    print("\n Portefeuille optimisé selon le ratio de Sharpe maximal :")
    for asset, w in zip(df_bl.columns, w_sharpe):
        print(f"{asset:<10}: {w:.4f}")
    print(f"  -> Rendement attendu : {ret_sharpe:.4%}")
    print(f"  -> Risque attendu    : {risk_sharpe:.4%}")
    print(f"  -> Sharpe Ratio      : {ret_sharpe / risk_sharpe:.2f}")

    # CVaR
    w_cvar, cvar_value = min_cvar_portfolio(df_bl.values, alpha=0.01)
    print("\n Portefeuille minimisant la CVaR à 1% :")
    for asset, w in zip(df_bl.columns, w_cvar):
        print(f"{asset:<10}: {w:.4f}")
    print(f"  -> CVaR (1%) estimée : {cvar_value:.4%}")

    # STARR
    w_starr, ret_starr, cvar_starr = max_starr_portfolio(df_bl.values, alpha=0.01)
    print("\n Portefeuille maximisant le ratio STARR (Sharpe/CVaR) :")
    for asset, w in zip(df_bl.columns, w_starr):
        print(f"{asset:<10}: {w:.4f}")
    print(f"  -> Rendement attendu : {ret_starr:.4%}")
    print(f"  -> CVaR estimée      : {cvar_starr:.4%}")
    print(f"  -> STARR Ratio       : {ret_starr / cvar_starr:.2f}")

    # Regrouper les poids dans un DataFrame
    weights_df = pd.DataFrame({
        "Max Sharpe": w_sharpe,
        "Min CVaR": w_cvar,
        "Max STARR": w_starr
    }, index=df_bl.columns)

    weights_df.to_csv("output/weights_optimisés.csv")
    print("\n Sauvegarde des poids optimisés dans 'output/weights_optimisés.csv' terminée.")

    print("\n Ouverture du rapport final contenant les visualisations graphiques.")
    pdf_path = os.path.abspath("output/Rapport_Visualisations.pdf")
    os.startfile(pdf_path)
if __name__ == "__main__":
    main()