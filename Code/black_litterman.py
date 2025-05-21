import numpy as np
import pandas as pd

def compute_equilibrium_return(cov_matrix, market_weights, delta=2.5):
    """
    Calcule les rendements d’équilibre à partir de la théorie du CAPM.

    Inputs:
        cov_matrix : matrice de covariance (Σ)
        market_weights : vecteur des pondérations de marché
        delta : aversion au risque (typ. 2 à 3)

    Output:
        pi : vecteur de rendements d’équilibre
    """
    pi = delta * cov_matrix @ market_weights
    return pi

def compute_posterior(pi, cov, P, q, tau=0.05, Omega=None):
    """
    Calcule les rendements et la covariance postérieures dans le modèle Black–Litterman.

    Inputs:
        pi : rendements d'équilibre
        cov : matrice de covariance Σ
        P : matrice des vues
        q : vecteur des vues
        tau : incertitude sur le marché
        Omega : matrice d'incertitude des vues (si None, auto : diag(P @ τΣ @ Pᵀ))

    Outputs:
        mu_post : rendements ajustés
        cov_post : covariance ajustée
    """
    tau_sigma = tau * cov

    if Omega is None:
        Omega = np.diag(np.diag(P @ tau_sigma @ P.T))

    middle = np.linalg.inv(P @ tau_sigma @ P.T + Omega)

    # ⚠️ correction ici : assure que pi est bien 2D colonne pour broadcast
    pi = pi.reshape(-1, 1)
    mu_post = pi + tau_sigma @ P.T @ middle @ (q - P @ pi)

    # 🧼 aplatir pour qu’il ait forme (n,)
    mu_post = mu_post.flatten()

    cov_post = cov + tau_sigma - tau_sigma @ P.T @ middle @ P @ tau_sigma

    return mu_post, cov_post

def generate_posterior_returns(mu_post, cov_post, n_sim=1000):
    """
    Génère des rendements simulés selon la distribution postérieure.

    Output : DataFrame (n_sim, n_assets)
    """
    mu_post = mu_post.flatten()  
    sim = np.random.multivariate_normal(mu_post, cov_post, size=n_sim)
    return sim
