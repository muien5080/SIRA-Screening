import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
torch.manual_seed(42)

DATA_PATH = "results/stochastic_dataset.npz"
MODEL_PATH = "results/sir_nn_model.pt"


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
            nn.Sigmoid()
        )

    def forward(self, t):
        return self.net(t)


def load_data():

    data = np.load(DATA_PATH)

    t = data["t"]
    S = data["S"]
    I = data["I"]
    R = data["R"]

    X = []
    Y = []

    for i in range(S.shape[0]):

        y = np.vstack([S[i], I[i], R[i]]).T

        X.append(t)
        Y.append(y)

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    return X, Y


def train_model(epochs=3000, lr=1e-3):

    t, y = load_data()

    t_norm = t / t.max()
    y_norm = y / 1000

    t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1,1)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)

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

    return model, t_tensor, y_tensor


def evaluate(model, t_tensor, y_tensor):

    with torch.no_grad():
        pred = model(t_tensor).numpy()

    y_true = y_tensor.numpy()

    mse = mean_squared_error(y_true, pred)
    r2 = r2_score(y_true, pred)

    print("MSE:", mse)
    print("R2:", r2)

    error = np.abs(y_true - pred).mean(axis=1)

    plt.figure()
    plt.plot(error)
    plt.title("Prediction Error vs Time")
    plt.xlabel("Time index")
    plt.ylabel("Mean Absolute Error")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/error_plot.png")

    print("Error plot saved")


def save_model(model):

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":

    model, t_tensor, y_tensor = train_model()

    evaluate(model, t_tensor, y_tensor)

    save_model(model)