import numpy as np
from utils import payoff

def monte_carlo_lsm(
    S0, K, T, r, sigma, option_type="call", position_type="buy",
    paths=50000, steps=100, return_paths=False
):
    """
    American Option Pricing using Longstaff-Schwartz Monte Carlo
    Returns price, and optionally simulated paths + boundary
    """
    dt = T / steps
    discount = np.exp(-r * dt)

    # Simulate price paths
    S = np.zeros((steps+1, paths))
    S[0] = S0
    for t in range(1, steps+1):
        z = np.random.standard_normal(paths)
        S[t] = S[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

    # Payoffs
    if option_type == "call":
        payoffs = np.maximum(S - K, 0)
    else:
        payoffs = np.maximum(K - S, 0)

    cashflows = payoffs[-1]
    boundary = []

    # Backward induction
    for t in range(steps-1, 0, -1):
        itm = payoffs[t] > 0
        if np.any(itm):
            X = S[t, itm]
            Y = cashflows[itm] * discount
            A = np.vstack([np.ones(X.shape), X, X**2]).T
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = coeffs[0] + coeffs[1]*X + coeffs[2]*X**2

            # Exercise decision
            exercise = payoffs[t, itm] > continuation
            cashflows[itm] = np.where(exercise, payoffs[t, itm], cashflows[itm] * discount)

            # Approximate boundary: min price where exercise happens
            if np.any(exercise):
                boundary.append((t*dt, np.min(X[exercise])))

        cashflows = cashflows * discount

    price = np.mean(cashflows)

    if return_paths:
        return price if position_type == "buy" else -price, S, boundary
    else:
        return price if position_type == "buy" else -price

def monte_carlo_boundary_avg(S0, K, T, r, sigma, option_type, position_type, runs=10, paths=5000, steps=100):
    boundaries = []

    for _ in range(runs):
        _, _, boundary = monte_carlo_lsm(
            S0, K, T, r, sigma, option_type, position_type,
            paths=paths, steps=steps, return_paths=True
        )
        if boundary:
            boundaries.append(boundary)

    # Average boundary over runs
    if boundaries:
        # Flatten and group by time
        all_points = {}
        for run in boundaries:
            for t, p in run:
                all_points.setdefault(t, []).append(p)

        avg_boundary = [(t, np.mean(prices)) for t, prices in sorted(all_points.items())]
        return avg_boundary
    return []

def calculate_greeks(
    S0, K, T, r, sigma, option_type="call", position_type="buy", paths=50000, steps=100
):
    eps = 1e-2  # perturbation for finite difference

    base_price = monte_carlo_lsm(S0, K, T, r, sigma, option_type, position_type, paths, steps)

    # Delta: dV/dS
    price_up = monte_carlo_lsm(S0 + eps, K, T, r, sigma, option_type, position_type, paths, steps)
    price_down = monte_carlo_lsm(S0 - eps, K, T, r, sigma, option_type, position_type, paths, steps)
    delta = (price_up - price_down) / (2 * eps)

    # Gamma: d²V/dS²
    gamma = (price_up - 2 * base_price + price_down) / (eps ** 2)

    # Vega: dV/dσ
    vega = (monte_carlo_lsm(S0, K, T, r, sigma + eps, option_type, position_type, paths, steps) - 
            monte_carlo_lsm(S0, K, T, r, sigma - eps, option_type, position_type, paths, steps)) / (2 * eps)

    # Theta: dV/dT (negative sign since time decay reduces value)
    theta = (monte_carlo_lsm(S0, K, T - eps, r, sigma, option_type, position_type, paths, steps) - 
             monte_carlo_lsm(S0, K, T + eps, r, sigma, option_type, position_type, paths, steps)) / (2 * eps)

    # Rho: dV/dr
    rho = (monte_carlo_lsm(S0, K, T, r + eps, sigma, option_type, position_type, paths, steps) - 
           monte_carlo_lsm(S0, K, T, r - eps, sigma, option_type, position_type, paths, steps)) / (2 * eps)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }

def binomial_tree(
    S0, K, T, r, sigma, option_type="call", position_type="buy", steps=200
):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Stock prices at maturity
    prices = np.array([S0 * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    
    # Payoff at maturity
    if option_type == "call":
        values = np.maximum(prices - K, 0)
    else:
        values = np.maximum(K - prices, 0)
    
    # Backward induction
    for i in range(steps-1, -1, -1):
        prices = prices[:i+1] / u
        values = np.exp(-r*dt) * (p * values[1:i+2] + (1-p) * values[0:i+1])
        
        # Early exercise
        if option_type == "call":
            exercise = np.maximum(prices - K, 0)
        else:
            exercise = np.maximum(K - prices, 0)
        values = np.maximum(values, exercise)
    
    price = values[0]
    return price if position_type == "buy" else -price

def finite_difference_pde(
    S0, K, T, r, sigma, option_type="call", position_type="buy", Smax_factor=3, M=200, N=200
):
    # Grid setup
    Smax = S0 * Smax_factor
    dS = Smax / M
    dt = T / N

    # Stock prices grid
    S = np.linspace(0, Smax, M+1)

    # Option payoff at maturity
    if option_type == "call":
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)

    # Coefficients
    alpha = 0.25 * dt * (sigma**2 * (np.arange(M+1)**2) - r * np.arange(M+1))
    beta = -dt * 0.5 * (sigma**2 * (np.arange(M+1)**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (np.arange(M+1)**2) + r * np.arange(M+1))

    # Matrix setup (Crank-Nicolson)
    A = np.zeros((M-1, M-1))
    B = np.zeros((M-1, M-1))
    for i in range(M-1):
        if i > 0:
            A[i, i-1] = -alpha[i+1]
            B[i, i-1] = alpha[i+1]
        A[i, i] = 1 - beta[i+1]
        B[i, i] = 1 + beta[i+1]
        if i < M-2:
            A[i, i+1] = -gamma[i+1]
            B[i, i+1] = gamma[i+1]

    # Time stepping backwards
    from numpy.linalg import solve
    for j in range(N):
        V[1:M] = solve(A, B @ V[1:M])
        # Early exercise
        if option_type == "call":
            V = np.maximum(V, S - K)
        else:
            V = np.maximum(V, K - S)

    # Interpolate to find S0
    price = np.interp(S0, S, V)
    return price if position_type == "buy" else -price
