import numpy as np
import os

np.random.seed(42)

N = 1000
BETA_VALUES = np.linspace(0.2, 0.5, 4)
GAMMA_VALUES = np.linspace(0.05, 0.2, 4)

T = 160
DT = 1


def simulate_sir(beta, gamma):

    S = np.zeros(T)
    I = np.zeros(T)
    R = np.zeros(T)

    S[0] = N - 1
    I[0] = 1
    R[0] = 0

    for t in range(1, T):

        infection_prob = beta * S[t-1] * I[t-1] / N
        recovery_prob = gamma * I[t-1]

        new_infections = np.random.poisson(infection_prob)
        new_recoveries = np.random.poisson(recovery_prob)

        new_infections = min(new_infections, S[t-1])
        new_recoveries = min(new_recoveries, I[t-1])

        S[t] = S[t-1] - new_infections
        I[t] = I[t-1] + new_infections - new_recoveries
        R[t] = R[t-1] + new_recoveries

    return S, I, R


def generate_dataset():

    all_S = []
    all_I = []
    all_R = []

    for beta in BETA_VALUES:
        for gamma in GAMMA_VALUES:

            S, I, R = simulate_sir(beta, gamma)

            all_S.append(S)
            all_I.append(I)
            all_R.append(R)

    all_S = np.array(all_S)
    all_I = np.array(all_I)
    all_R = np.array(all_R)

    t = np.arange(T)

    os.makedirs("results", exist_ok=True)

    np.savez(
        "results/stochastic_dataset.npz",
        t=t,
        S=all_S,
        I=all_I,
        R=all_R
    )

    print("Dataset saved: results/stochastic_dataset.npz")


if __name__ == "__main__":
    generate_dataset()