# SIRA Screening - GSoC 2026

## Overview
This repository contains the **SIRA (Susceptible-Infected-Removed Approximation) screening project** for CERN’s Human-AI initiative.  

The main goals are:  
1. Simulate epidemic data using the **stochastic SIR model**.  
2. Train **machine learning models** to predict mean counts of `S(t)`, `I(t)`, `R(t)`.  
3. Use **symbolic ML methods** to approximate the deterministic SIR equations.

This project serves as a **screening exercise** for the upcoming GSoC 2026 application.

---

## Repository Structure
SIRA-screening/
├── notebooks/
│   └── sir_experiments.ipynb
├── results/              # (currently empty or not added)
├── src/
│   ├── deterministic_sir.py
│   ├── ml_model.py
│   └── stochastic_sir.py
├── .gitignore
├── README.md
└── requirements.txt

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/SIRA-screening.git
   cd SIRA-screening

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt


