# SIRA Screening Project – Epidemic Modeling with SIR and Machine Learning

This repository implements a small research-style pipeline for simulating and analyzing epidemic dynamics using the **SIR (Susceptible–Infected–Recovered)** model.

The project combines three components:

1. A **stochastic epidemic simulator**
2. A **deterministic SIR model**
3. A **machine learning model** that learns epidemic trajectories from simulation data

The goal is to demonstrate how **simulation data can be used to train a neural network that approximates the evolution of an epidemic over time.**

---

# Project Overview

The workflow of the project is:

1. Generate epidemic simulations using a **stochastic SIR model**
2. Run a **deterministic SIR model** for comparison
3. Train a **neural network** to learn the mapping
    t → (S(t), I(t), R(t))

4. Evaluate the model using **Mean Squared Error (MSE)** and **R² score**
5. Visualize prediction errors and epidemic trajectories

---

# Repository Structure


```
SIRA-screening/
│
├── notebooks/
│   └── sir_experiments.ipynb        # Main experiment notebook
│
├── src/
│   ├── stochastic_sir.py            # Poisson-distributed infection and recovery events
│   ├── deterministic_sir.py         # Deterministic ODE SIR model
│   └── ml_model.py                  # Neural network training
│
├── results/
│   ├── stochastic_means.npz         # Mean stochastic trajectories
│   ├── deterministic_sir.npz        # Deterministic solution
│   └── sir_nn_model.pt              # Trained neural network
│
├── requirements.txt
└── README.md
```
---


---

# SIR Epidemic Model

The SIR model divides the population into three compartments:

| Variable | Meaning |
|--------|--------|
| **S(t)** | Susceptible population |
| **I(t)** | Infected population |
| **R(t)** | Recovered population |

The deterministic SIR equations are:

```
dS/dt = -β S I / N
dI/dt = β S I / N − γ I
dR/dt = γ I
```


Where:

- **β** – infection rate  
- **γ** – recovery rate  
- **N** – total population  

---

## 1. Stochastic Epidemic Simulation

File: `src/stochastic_sir.py`


The stochastic simulator generates epidemic trajectories using **Poisson-distributed infection and recovery events**.

Multiple epidemics are simulated across different parameter values:
β ∈ [0.2, 0.5]
γ ∈ [0.05, 0.2]

The dataset is saved as:

File: `results/stochastic_dataset.npz`

Stored variables:

t : time points
S : susceptible trajectories
I : infected trajectories
R : recovered trajectories

---

## 2. Deterministic SIR Model

File: `src/deterministic_sir.py`


This script simulates the deterministic SIR equations using discrete time updates.

Output:

File:   `results/deterministic_sir.npz`


---

## 3. Machine Learning Model

File: `src/ml_model.py`

A neural network is trained to predict the epidemic dynamics.

### Model Architecture
Input: time t

Neural Network:
1 → 128 → 128 → 64 → 3

Activation:
ReLU hidden layers
Sigmoid output

The network predicts:
[S(t), I(t), R(t)]



The trained model is saved as:
File: `results/sir_nn_model.pt`



---

# Evaluation Metrics

Model performance is evaluated using:

- **Mean Squared Error (MSE)**
- **R² score**

The project also generates a diagnostic visualization:

Prediction Error vs Time

Saved as:
File: `results/error_plot.png`


---

# Installation

Clone the repository:

```bash
git clone https://github.com/muien5080/SIRA-Screening.git
cd SIRA-Screening
```
Install dependencies:
```bash
pip install -r requirements.txt
```

Running the Pipeline

Generate stochastic epidemic dataset:
```bash
python src/stochastic_sir.py
```

Run deterministic SIR model:
```bash
python src/deterministic_sir.py
```

Train the neural network model:
```bash
python src/ml_model.py
```

Running Experiments

Open the notebook:

File: `notebooks/experiments.ipynb`

Run all cells to:
generate epidemic simulations
train the neural network
visualize epidemic trajectories
evaluate prediction accuracy

Example Outputs:
The project produces,
stochastic epidemic trajectories
deterministic SIR curves
neural network predictions of epidemic dynamics
prediction error plots
