# src/stochastic_sir.py
import numpy as np
import os

def run_stochastic_sir(beta=0.3, gamma=0.1, N=1000, I0=10, t_max=160, dt=1, num_runs=50, save_results=False, random_seed=None):
    """
    Run multiple stochastic Gillespie SIR simulations and compute mean trajectories.
    
    Args:
        beta (float): Infection rate
        gamma (float): Recovery rate
        N (int): Total population
        I0 (int): Initial number of infected individuals
        t_max (float): Maximum simulation time
        dt (float): Time step for interpolation
        num_runs (int): Number of stochastic simulations
        save_results (bool): Whether to save the results to disk
        random_seed (int or None): Random seed for reproducibility
    
    Returns:
        time_grid (np.array): Fixed time grid
        S_mean (np.array): Mean susceptible over runs
        I_mean (np.array): Mean infected over runs
        R_mean (np.array): Mean recovered over runs
    """
    
    # --- Input validation ---
    if num_runs <= 0:
        raise ValueError("Number of runs must be positive!")
    if beta <= 0 or gamma <= 0:
        raise ValueError("Infection and recovery rates must be positive!")
    if I0 <= 0 or I0 >= N:
        raise ValueError("Initial infected I0 must be greater than 0 and less than total population N!")
    
    # Set the random seed for reproducibility (if provided)
    if random_seed is not None:
        np.random.seed(random_seed)

    # --- State change vectors ---
    v1 = np.array([-1, +1, 0])  # Infection: S -> I
    v2 = np.array([0, -1, +1])  # Recovery: I -> R

    # Time grid for interpolation
    time_grid = np.arange(0, t_max + dt, dt)
    
    # Arrays to store all runs
    S_all = np.zeros((num_runs, len(time_grid)))
    I_all = np.zeros((num_runs, len(time_grid)))
    R_all = np.zeros((num_runs, len(time_grid)))

    def Gillespie_loop(state, current_time):
        t_values = [current_time]
        S_values = [state[0]]
        I_values = [state[1]]
        R_values = [state[2]]

        while state[1] > 0:
            S = state[0]
            I = state[1]
            a1 = beta * S * I / N
            a2 = gamma * I
            a0 = a1 + a2

            tau = np.random.exponential(1 / a0)
            current_time += tau

            r = np.random.rand()
            if r < a1 / a0:
                state = state + v1
            else:
                state = state + v2

            t_values.append(current_time)
            S_values.append(state[0])
            I_values.append(state[1])
            R_values.append(state[2])

        return np.array(t_values), np.array(S_values), np.array(I_values), np.array(R_values)

    # --- Run multiple stochastic simulations ---
    for run in range(num_runs):
        X0 = np.array([N - I0, I0, 0])  # Initial state
        t0 = 0  # Start time
        t_vals, S_vals, I_vals, R_vals = Gillespie_loop(X0, t0)

        # Interpolate to fixed time grid
        S_all[run, :] = np.interp(time_grid, t_vals, S_vals)
        I_all[run, :] = np.interp(time_grid, t_vals, I_vals)
        R_all[run, :] = np.interp(time_grid, t_vals, R_vals)

    # Compute mean values across runs
    S_mean = S_all.mean(axis=0)
    I_mean = I_all.mean(axis=0)
    R_mean = R_all.mean(axis=0)

    # --- Save results (if specified) ---
    if save_results:
        os.makedirs('results', exist_ok=True)
        np.savez('results/stochastic_means.npz', t=time_grid, S=S_mean, I=I_mean, R=R_mean)

    # Return the time grid and mean values
    return time_grid, S_mean, I_mean, R_mean


# Example execution (only when the script is executed directly)
if __name__ == "__main__":
    try:
        # Run the stochastic SIR simulation
        time_grid, S_mean, I_mean, R_mean = run_stochastic_sir(beta=0.3, gamma=0.1, N=1000, I0=10, t_max=160, dt=1, num_runs=50, save_results=True, random_seed=42)

        # Display the final mean values
        print(f"Stochastic SIR simulation complete. Final mean values:")
        print(f"Mean Susceptible: {S_mean[-1]}")
        print(f"Mean Infected: {I_mean[-1]}")
        print(f"Mean Recovered: {R_mean[-1]}")

    except Exception as e:
        print(f"Error: {e}")