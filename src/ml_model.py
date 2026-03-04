import numpy as np
import torch
import torch.nn as nn
import os

# Neural Network Architecture
class SIRNet(nn.Module):
    def __init__(self):
        super(SIRNet,self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(1,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,3)
        )
    
    def forward(self,t):
        return self.net(t)
    
# Load Stochastic simulation Data

def load_data(path="results/stochastic_means.npz"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Run stochastic_sir.py first to generate stochastic_means.npz"
        )
    
    data = np.load(path)
    t = data["t"]
    S = data["S"]
    I = data["I"]
    R = data["R"]
    y = np.vstack([S,I,R]).T
    return t,y

# Training Function

def train_model(beta = 0.3, gamma = 0.1, epochs = 5000, lr = 1e-3):
    t, y = load_data
    
    # convert to torch tensors
    t = torch.tensor(t,dtype=torch.float32).view(-1,1)
    y = torch.tensor(y,dtype=torch.float32)
    t.requires_grad(True)
    
    model = SIRNet
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    mse = nn.MSELoss()
    
    for epoch in range (epochs):
        pred = model(t)
        S_pred = pred [:,0]
        I_pred = pred [:,1]
        R_pred = pred [:,2]
        
        # Data Loss
        data_loss = mse(pred,y)
        
        # Physics Loss
        dS_dt = torch.autograd.grad(S_pred.sum(),t,create_graph=True)[0]
        dI_dt = torch.autograd.grad(I_pred.sum(),t,create_graph=True)[0]
        dR_dt = torch.autograd.grad(R_pred.sum(),t,create_graph=True)[0]
        
        physics_S = dS_dt + beta * S_pred * I_pred / 1000
        physics_I = dI_dt - beta * S_pred * I_pred / 1000 + gamma * I_pred
        physics_R = dR_dt - gamma * I_pred
        
        physics_loss = (
            mse(physics_S, torch.zeros_like(physics_S)) + 
            mse(physics_I, torch.zeros_like(physics_I)) + 
            mse(physics_R, torch.zeros_like(physics_R))
        )
        
        loss = data_loss + physics_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch%500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
            
        return model
    
# Save Trained Model
def save_model(model):
    os.makedirs("results",exist_ok=True)
    torch.save(model.state_dict(), "results/sir_pinn_model.pt")
    print("Model saved to results/sir_pinn_model.pt")
    
# Run Training
if __name__ == "__main__":
    model = train_model()
    save_model(model)