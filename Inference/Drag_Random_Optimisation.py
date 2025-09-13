import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from Models.GenNet_skip_v2 import AutoencoderSDF
import json
from sklearn.decomposition import PCA

# === CONFIGURATION ==
model_path = 'Your_Model_Weights.pt'
latent_path = 'Your_Latent_Train_path.npz'


with open('Your_rep/Cd_stats.json', 'r') as f:
    cd_stats = json.load(f)

Cd_min = cd_stats['Cd_min']
Cd_max = cd_stats['Cd_max']

def denormalize(Cd, Cd_max, Cd_min): #denormalisation of drag coefficient
  return Cd * (Cd_max - Cd_min) + Cd_min

latent_dim = 128
hidden_dim = 256
dropout = 0.05
epsilon = 0.1  # noise std
n_repeat = 10  # quantity of new vectors for one initial vectors 
MC = 100  # Monte Carlo amount of calculs (the more the better)

device = torch.device('cpu')

# === MODEL LOADING ===
model = AutoencoderSDF(latent_dim, hidden_dim, dropout).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() 

# === MC DROPOUT ===

def enable_dropout(m):
    for module in m.modules():
        if isinstance(module, nn.Dropout):
            module.train()

enable_dropout(model)

# === INITIAL LATENT VECTORS LOADING ===
z = np.load(latent_path, allow_pickle=True)['z']  # shape (N, latent_dim)

# PCA on z to obtain composants variance
pca = PCA(n_components=latent_dim)
pca.fit(z)
variances = pca.explained_variance_  # shape: (latent_dim,)

alpha = 0.005

# === GÉNÉRATION DES NOUVEAUX Z ===
z_repeated = np.repeat(z, n_repeat, axis=0)  # shape: (N * n_repeat, latent_dim)
noise = np.random.randn(*z_repeated.shape) * np.sqrt((alpha * variances))  # broadcasting applied

z_new = z_repeated + noise
z_new = torch.tensor(z_new, dtype=torch.float32).to(device)

# === CALCUL OF UNCERTAINTY WITH MC DROPOUT ===
def get_uncertainty(z_single, T):
    preds = []
    for _ in range(T):
        with torch.no_grad():
            y_pred = denormalize(model.predict_cd_only(z_single), Cd_max, Cd_min)  # shape (1, 1)
        preds.append(y_pred.cpu().item())  # scalar
    preds = np.array(preds)
    return preds.mean(), preds.std()

mean_preds, std_preds = [], []

for i in tqdm(range(len(z_new))):
    z_i = z_new[i].unsqueeze(0)  # (1, latent_dim)
    mu, sigma = get_uncertainty(z_i, MC)
    mean_preds.append(mu)
    std_preds.append(sigma)

Z = []
preds = []

for i in range(len(mean_preds)):
    if mean_preds[i] < Cd_min:
        Z.append(z_new[i]) #selection of latent that generate better results than best drag coefficient in the database
        preds.append(mean_preds[i])

Z_path = 'Z_optim.npz'
np.savez(Z_path, latent = Z, predictions = preds)
print(len(Z))



