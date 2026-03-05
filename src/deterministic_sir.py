import numpy as np
from scipy.integrate import odeint
import os


def sir_ode(y, t, beta, gamma, N):
    """
    SIR differential equations.
    
    dS/dt = -beta * S * I / N
    dI/dt = beta * S * I / N - gamma * I
    dR/dt = gamma * I
    """

    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def run_deterministic_sir(beta=0.3, gamma=0.1, N=1000, I0=10, t_max=160, save_results=False):
    """
    Run deterministic SIR simulation.
    Arguments:
        beta (float): infection rate
        gamma (float): recovery rate
        N (int): total population
        I0 (int): initial infected
        t_max (int): simulation time
        save_results (bool): save results to disk

    Returns:
        t, S, I, R
    """

    if I0 <= 0 or I0 >= N:
        raise ValueError("I0 must be between 0 and N")

    S0 = N - I0
    R0 = 0
    y0 = [S0, I0, R0]
    t = np.linspace(0, t_max, t_max + 1)
    sol = odeint(sir_ode, y0, t, args=(beta, gamma, N))
    S, I, R = sol.T

    if save_results:
        os.makedirs("results", exist_ok=True)
        np.savez("results/deterministic_sir.npz", t=t, S=S, I=I, R=R)

    return t, S, I, R


if __name__ == "__main__":
    t, S, I, R = run_deterministic_sir(save_results=True)
    print("Deterministic SIR simulation complete")
    print("Final values:")
    print("S:", S[-1])
    print("I:", I[-1])
    print("R:", R[-1])
    
