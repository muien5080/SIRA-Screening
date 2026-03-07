import numpy as np
import os

N = 1000
beta = 0.3
gamma = 0.1

T = 160
dt = 1


def simulate():

    S = np.zeros(T)
    I = np.zeros(T)
    R = np.zeros(T)

    S[0] = N - 1
    I[0] = 1
    R[0] = 0

    for t in range(1, T):

        dS = -beta * S[t-1] * I[t-1] / N
        dI = beta * S[t-1] * I[t-1] / N - gamma * I[t-1]
        dR = gamma * I[t-1]

        S[t] = S[t-1] + dS
        I[t] = I[t-1] + dI
        R[t] = R[t-1] + dR

    return S, I, R


if __name__ == "__main__":

    S, I, R = simulate()

    os.makedirs("results", exist_ok=True)

    np.savez(
        "results/deterministic_sir.npz",
        S=S,
        I=I,
        R=R
    )

    print("Deterministic simulation saved")