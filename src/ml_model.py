import numpy as np
import torch
import torch.nn as nn
import os

# -----------------------------
# constants
# -----------------------------
N = 1000
DATA_PATH = "results/stochastic_means.npz"
MODEL_PATH = "results/sir_nn_model.pt"


# -----------------------------
# neural network
# -----------------------------
class SIRNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,3),
            nn.Sigmoid()   # ensures outputs in [0,1]
        )

    def forward(self,t):
        return self.net(t)


# -----------------------------
# load stochastic data
# -----------------------------
def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Run stochastic_sir.py first to generate stochastic_means.npz"
        )

    data = np.load(DATA_PATH)

    t = data["t"]
    S = data["S"]
    I = data["I"]
    R = data["R"]

    y = np.vstack([S,I,R]).T

    return t,y


# -----------------------------
# training
# -----------------------------
def train_model(epochs=5000, lr=1e-3):

    t, y = load_data()

    # normalize
    t_norm = t / t.max()
    y_norm = y / N

    t_tensor = torch.from_numpy(t_norm.astype(np.float32)).view(-1,1)
    y_tensor = torch.from_numpy(y_norm.astype(np.float32))

    model = SIRNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        pred = model(t_tensor)

        loss = loss_fn(pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss {loss.item():.6f}")

    return model


# -----------------------------
# save model
# -----------------------------
def save_model(model):

    os.makedirs("results", exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)

    print("Model saved to", MODEL_PATH)


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":

    model = train_model()

    save_model(model)