# SIRA Screening Project – Epidemic Modeling with SIR and Machine Learning

This repository implements a small research-style pipeline for simulating and analyzing epidemic dynamics using the **SIR (Susceptible–Infected–Recovered)** model.

The project combines three components:

1. A **stochastic epidemic simulator** using the Gillespie algorithm
2. A **deterministic SIR model** solved with ordinary differential equations
3. A **machine learning model** that learns to predict the mean epidemic trajectories

The goal is to demonstrate how simulation data can be used to train a neural network that approximates the evolution of an epidemic over time.

---

## Project Overview

The workflow follows these steps:

1. Simulate an epidemic using a **stochastic SIR model**.
2. Run a **deterministic SIR model** for comparison.
3. Compute the **mean epidemic trajectory** from multiple stochastic runs.
4. Train a **neural network** to learn the mapping
   t → (S(t), I(t), R(t))


5. Compare ML predictions with both stochastic and deterministic results.

---

## Repository Structure
SIRA-screening/
│
├── notebooks/
│ └── sir_experiments.ipynb # Main experiment notebook
│
├── src/
│ ├── stochastic_sir.py # Gillespie stochastic simulation
│ ├── deterministic_sir.py # Deterministic ODE SIR model
│ └── ml_model.py # Neural network training
│
├── results/
│ ├── stochastic_means.npz # Mean stochastic trajectories
│ ├── deterministic_sir.npz # Deterministic solution
│ └── sir_nn_model.pt # Trained neural network
│
├── requirements.txt
└── README.md


---

## SIR Model

The classical SIR model divides the population into three compartments:

- **S(t)** – Susceptible individuals
- **I(t)** – Infected individuals
- **R(t)** – Recovered individuals

The deterministic equations are:
dS/dt = -β S I / N
dI/dt = β S I / N − γ I
dR/dt = γ I


Where:

- **β** – infection rate  
- **γ** – recovery rate  
- **N** – total population  

---

## 1. Stochastic Epidemic Simulation

File: `src/stochastic_sir.py`

The stochastic epidemic is simulated using the **Gillespie algorithm**.

Two events are possible:

| Event | State Change |
|------|-------------|
| Infection | S → I |
| Recovery | I → R |

Multiple simulation runs are performed and the **mean trajectories** are computed.

Output saved to:
'results/stochastic_means.npz'


---

## 2. Deterministic SIR Model

File: `src/deterministic_sir.py`

The deterministic model solves the SIR differential equations using `scipy.integrate.odeint`.

This provides a smooth approximation of the epidemic dynamics, which can be compared to the stochastic mean.

Output saved to:
   results/deterministic_sir.npz


---

## 3. Machine Learning Model

File: `src/ml_model.py`

A neural network is trained to predict the epidemic trajectories.

### Model Architecture
Input: time t

Neural Network:
1 → 128 → 128 → 64 → 3

Activation:
ReLU hidden layers
Sigmoid output

The network predicts:
[S(t), I(t), R(t)]


normalized by population size.

Training uses **Mean Squared Error (MSE)** between predictions and stochastic mean trajectories.

The trained model is saved as:
results/sir_nn_model.pt


---

## Notebook Experiments

The notebook `notebooks/sir_experiments.ipynb` demonstrates the full workflow:

1. Load stochastic simulation results
2. Plot S, I, R epidemic curves
3. Compare stochastic vs deterministic models
4. Load the trained neural network
5. Generate ML predictions
6. Compare predictions with simulation data

Example outputs include:

- Mean stochastic epidemic curves
- Stochastic vs deterministic comparison
- ML prediction vs true epidemic trajectory

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/SIRA-screening.git
cd SIRA-screening

Install dependencies:
   pip install -r requirements.txt

Running the Pipeline
1. Run stochastic simulations
python src/stochastic_sir.py
2. Run deterministic model
python src/deterministic_sir.py
3. Train the ML model
python src/ml_model.py
4. Explore results

Open the notebook:

notebooks/sir_experiments.ipynb

## AI Assistance Disclosure

Parts of the documentation and debugging process for this project were assisted using **ChatGPT (OpenAI)**.

Specifically, ChatGPT was used to:

- Help debug issues in the machine learning training pipeline
- Verify the correctness of stochastic and deterministic SIR implementations
- Assist in writing and formatting this README file
- Suggest improvements to code organization and reproducibility

All implementation decisions, simulations, and experiments were designed, implemented, and validated by the author.

The use of AI assistance was limited to **code explanation, documentation support, and debugging guidance**, and the final repository structure and results were verified manually.