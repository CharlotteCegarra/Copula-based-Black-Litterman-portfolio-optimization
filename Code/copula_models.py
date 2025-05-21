from copulas.bivariate import Clayton, Gumbel, Frank
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pyvinecopulib import Vinecop, BicopFamily, FitControlsVinecop

def fit_bivariate_copula(u1, u2, copula_type="clayton"):
    """
    Ajuste une copule bivariée (Clayton, Gumbel, Frank) sur deux séries de pseudo-observations.
    """
    data = pd.DataFrame({'u1': u1, 'u2': u2}).dropna()

    if copula_type == "clayton":
        copula = Clayton()
    elif copula_type == "gumbel":
        copula = Gumbel()
    elif copula_type == "frank":
        copula = Frank()
    else:
        raise ValueError(f"Copule non supportée : {copula_type}")

    copula.fit(data.values)
    return copula

def plot_copula_sample(copula, n_samples=1000):
    """
    Affiche un nuage de points simulé à partir d'une copule ajustée.
    """
    samples = copula.sample(n_samples)
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title(f"Échantillon simulé ({type(copula).__name__})")
    plt.xlabel("u1 (simulé)")
    plt.ylabel("u2 (simulé)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_real_vs_simulated(u1, u2, copula, copula_name=""):
    """
    Compare les vraies données (pseudo-observations) avec un échantillon simulé de la copule ajustée.
    """
    simulated = copula.sample(1000)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Vraies données
    axs[0].scatter(u1, u2, alpha=0.5, color='blue')
    axs[0].set_title("Vraies pseudo-observations")
    axs[0].set_xlabel("u1 (LVMH)")
    axs[0].set_ylabel("u2 (Sanofi)")
    axs[0].grid(True)

    # Simulées
    axs[1].scatter(simulated[:, 0], simulated[:, 1], alpha=0.5, color='green')
    axs[1].set_title(f"Simulées par copule {copula_name}")
    axs[1].set_xlabel("u1 (simulé)")
    axs[1].set_ylabel("u2 (simulé)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def fit_vine_copula(u_data):
    """
    Ajuste une Vine Copula multivariée à partir des pseudo-observations [0,1]^d.
    Compatible avec les versions récentes de pyvinecopulib.
    """
    u = np.asfortranarray(u_data.values)  # Format attendu par le backend C++

    control = FitControlsVinecop(
        family_set=[
            BicopFamily.clayton,
            BicopFamily.frank,
            BicopFamily.gumbel,
            BicopFamily.gaussian,
            BicopFamily.student
        ],
        selection_criterion="mbicv",
        trunc_lvl=0
    )

    vine = Vinecop(d=u.shape[1])
    vine.select(data=u, controls=control)

    return vine

def simulate_joint_returns(vine_copula, sigmas_dict, residuals_dict, n_sim=1000):
    """
    Simule des rendements multivariés conditionnels à partir :
    - d'une vine copula ajustée
    - des modèles marginaux (GARCH → résidus & volatilités conditionnelles)

    Inputs :
        vine_copula : copule vine ajustée sur les pseudo-observations
        sigmas_dict : dict des séries sigma_t (volatilités conditionnelles) par actif
        residuals_dict : dict des résidus GARCH par actif
        n_sim : nombre de simulations à générer

    Output :
        DataFrame des rendements simulés de taille (n_sim, nb_actifs)
    """
    # Étape 1 : simulation dans l’espace [0,1]^d
    u_sim = vine_copula.simulate(n_sim)  # shape (n_sim, d)
    tickers = list(sigmas_dict.keys())

    # Étape 2 : inversion des marges avec quantile normal (on suppose résidus ~ N(0,1))
    z_sim = norm.ppf(u_sim)

    # Étape 3 : reconstruction des rendements simulés : r = z * sigma_t
    last_sigmas = np.array([sigmas_dict[ticker].iloc[-1] for ticker in tickers])
    returns_sim = z_sim * last_sigmas  # broadcast

    return pd.DataFrame(returns_sim, columns=tickers)

def fit_copula_clayton(u_data):
    from pyvinecopulib import Vinecop, BicopFamily, FitControlsVinecop
    import numpy as np
    u = np.asfortranarray(u_data)
    control = FitControlsVinecop(
        family_set=[BicopFamily.clayton],
        selection_criterion="mbicv"
    )
    vine = Vinecop(d=u.shape[1])
    vine.select(u, controls=control)
    return vine

def fit_copula_student(u_data):
    from pyvinecopulib import Vinecop, BicopFamily, FitControlsVinecop
    import numpy as np
    u = np.asfortranarray(u_data)
    control = FitControlsVinecop(
        family_set=[BicopFamily.student],
        selection_criterion="mbicv"
    )
    vine = Vinecop(d=u.shape[1])
    vine.select(u, controls=control)
    return vine
