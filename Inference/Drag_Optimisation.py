from Models.GenNet_skip import AutoencoderSDF
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import yaml
import torch.nn as nn
import json
from scipy.spatial.distance import cdist

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the latent space created by your model. You can easily got it by inferencing the decoder of the model only with train/validation or test data (SDF).
# Loading latent train

train_path = 'Your_Latent_Train_Path.npz'
latent_train = np.load(train_path, allow_pickle=True) #contains all latent vectors corresponding to training data
z_train = latent_train['z']
Z_train = torch.tensor(z_train, dtype=torch.float32, device=device)  # shape (N, D)

ids = latent_train['shape_id']

# Calcul mean and std
mu_train = np.mean(z_train, axis=0)
cov_train = np.cov(z_train, rowvar=False)
eps = 1e-6
cov_train += eps * np.eye(cov_train.shape[0])
cov_inv = np.linalg.inv(cov_train)
cov_inv_torch = torch.tensor(cov_inv, dtype=torch.float32, device=device)


# calcul of <d> (average min distance between vectors of the latent space of train)
Dmat = cdist(z_train, z_train, metric='mahalanobis', VI=cov_inv)
np.fill_diagonal(Dmat, np.inf)
dmins = Dmat.min(axis=1)
d_mean = dmins.mean()
exp_d_mean = np.exp(d_mean)
exp_d_mean = torch.tensor(exp_d_mean, dtype=torch.float32, device=device)


# Load config
with open("/home/amb/bjacques/GenNet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open('/home/amb/bjacques/GenNet/data_split/cd_stats.json', 'r') as f:
    cd_stats = json.load(f)

Cd_min = cd_stats['Cd_min']
Cd_max = cd_stats['Cd_max']

def denormalize(Cd, Cd_max, Cd_min):
    return Cd * (Cd_max - Cd_min) + Cd_min

#be sure you use the propers hyperparameters
latent_dim = 128
hidden_dim = 256
dropout = 0.05 #Use the same value as for model training

# Load model
model = AutoencoderSDF(latent_dim, hidden_dim, dropout)
model.load_state_dict(torch.load("Your_Model_Weights.pt"))
model.to(device)

# Enable MC Dropout
def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

enable_dropout(model)
model.train()

# Init z
Z = z_train
N = len(Z)
z_init = None
for i in range(N):
    if ids[i][0] == 'F':
        z_init = Z[i]
        break

if z_init is None:
    raise ValueError("No latent vector with 'F' IF found.")

z = torch.tensor(z_init, dtype=torch.float32, requires_grad=True, device=device)
#Initialisation with fastback car shape

# Optim settings
lr = 1e-5
epochs = 1000
MC = 100 #Amount of MC dropout calculs (the more the better)
lambda_reg = 1 #parameter for gradient descent regularization using uncertainty measured by MC dropout
lambda_maha = 0.005 #parameter for gradient descent regularization using Mahalanobis distance from the center of the training manifold
optimizer = torch.optim.Adam([z], lr=lr)

Result = {
    'z_history': [],
    'epochs': [],
    'mu': [],
    'std': [],
    'Cd_pred': [],
    'norm': [], 
    'dist':[],
    'maha_reg':[]
}

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    preds = []
    delta = Z_train - z.unsqueeze(0)  # (N, D)
    dists_squared = torch.einsum('nd,dk,nk->n', delta, cov_inv_torch, delta)  # (N,)
    d_maha = torch.sqrt(dists_squared + 1e-6).min()  # scalaire torch
    maha_reg = torch.exp(d_maha) - exp_d_mean  # exp_d_mean reste float

    Result['dist'].append(d_maha.item())
    Result['maha_reg'].append(maha_reg.detach().cpu().item())

    # MC Dropout
    for _ in range(MC):
        Cd_pred = model.predict_cd_only(z)
        preds.append(Cd_pred)

    preds = torch.stack(preds, dim=0)
    mu = preds.mean()
    std = preds.std() * 10

    loss = mu + lambda_reg * std + lambda_maha * maha_reg
    loss.backward()
    optimizer.step()

    # Logging
    Result['z_history'].append(z.detach().cpu().clone().numpy())
    Result['epochs'].append(epoch)
    Result['mu'].append(denormalize(mu.item(), Cd_max, Cd_min))
    Result['std'].append(denormalize(std.item(), Cd_max - Cd_min, 0))
    Result['Cd_pred'].append(denormalize(mu.item(), Cd_max, Cd_min))
    Result['norm'].append(torch.norm(z).item())

# Save results
np.savez(f"Your_Results_path.npz",
         z_array=np.array(Result['z_history']),
         cd_array=np.array(Result['Cd_pred']),
         norm_array=np.array(Result['norm']),
         epoch_array=np.array(Result['epochs']),
         mu_array=np.array(Result['mu']),
         std_array=np.array(Result['std']),
         distance = np.array(Result['dist']),
         maha_reg = np.array(Result['maha_reg']))

print("file saved.")
