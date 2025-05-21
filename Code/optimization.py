import cvxpy as cp
import numpy as np

def max_sharpe_portfolio(returns, risk_aversion=10):
    """
    Maximisation approchée du Sharpe Ratio (μᵗw - λ·wᵗΣw), DCP-compliant.
    """
    n_assets = returns.shape[1]
    mu = np.mean(returns, axis=0)
    cov = np.cov(returns.T)

    w = cp.Variable(n_assets)
    ret = mu @ w
    risk = cp.quad_form(w, cov)

    # Objectif : trade-off entre rendement et risque
    objective = cp.Maximize(ret - risk_aversion * risk)
    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value, ret.value, cp.sqrt(risk).value


def min_cvar_portfolio(returns, alpha=0.01):
    """
    Minimise la CVaR empirique à partir de rendements simulés.
    Contraintes :
        - Long-only
        - Somme des poids = 1
    """
    n, d = returns.shape
    w = cp.Variable(d)
    VaR = cp.Variable()
    z = cp.Variable(n)

    portfolio_returns = returns @ w

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        z >= 0,
        z >= -portfolio_returns - VaR
    ]

    cvar = VaR + (1 / (alpha * n)) * cp.sum(z)
    prob = cp.Problem(cp.Minimize(cvar), constraints)
    prob.solve()

    return w.value, cvar.value


def max_starr_portfolio(returns, alpha=0.01, lambda_cvar=10):
    """
    Approximation du STARR ratio via une fonction DCP-compatible :
    maximise (mean - λ · CVaR)

    Contraintes :
        - Long-only
        - Somme des poids = 1
    """
    n, d = returns.shape
    w = cp.Variable(d)
    VaR = cp.Variable()
    z = cp.Variable(n)

    # Espérance de rendement
    mean_return = cp.sum(returns @ w) / n

    # Contraintes CVaR classiques
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        z >= 0,
        z >= -returns @ w - VaR
    ]

    cvar = VaR + (1 / (alpha * n)) * cp.sum(z)

    # Objectif DCP-compatible : rendement - λ × CVaR
    starr_proxy = mean_return - lambda_cvar * cvar
    prob = cp.Problem(cp.Maximize(starr_proxy), constraints)
    prob.solve()

    return w.value, mean_return.value, cvar.value
