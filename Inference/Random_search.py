import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from Models.GenNet_skip_v2 import AutoencoderSDF
import json
from sklearn.decomposition import PCA

# === CONFIGURATION ==
model_path = '/home/amb/bjacques/GenNet/Weights/Newmodel_1.pt'
latent_path = '/home/amb/bjacques/GenNet/Results/latent_train.npz'


with open('/home/amb/bjacques/GenNet/data_split/cd_stats.json', 'r') as f:
    cd_stats = json.load(f)

Cd_min = cd_stats['Cd_min']
Cd_max = cd_stats['Cd_max']

def denormalize(Cd, Cd_max, Cd_min): #dénormalise une valeur Cd
  return Cd * (Cd_max - Cd_min) + Cd_min

latent_dim = 128
hidden_dim = 256
dropout = 0.05
epsilon = 0.1  # écart-type du bruit
n_repeat = 10  # nombre de nouveaux z par z initial
MC = 1  # nombre de passes Monte Carlo

device = torch.device('cpu')

# === CHARGEMENT DU MODÈLE ===
model = AutoencoderSDF(latent_dim, hidden_dim, dropout).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # important, pour ne pas activer d'autres modules parasites

# === ACTIVATION DU DROPOUT POUR MC DROPOUT ===
"""
def enable_dropout(m):
    for module in m.modules():
        if isinstance(module, nn.Dropout):
            module.train()

enable_dropout(model)
"""


# === CHARGEMENT DES Z INITIAUX ===
z = np.load(latent_path, allow_pickle=True)['z']  # shape (N, latent_dim)

# PCA sur z pour obtenir les variances des composantes
pca = PCA(n_components=latent_dim)
pca.fit(z)
variances = pca.explained_variance_  # shape: (latent_dim,)

alpha = 100.0

# === GÉNÉRATION DES NOUVEAUX Z ===
z_repeated = np.repeat(z, n_repeat, axis=0)  # shape: (N * n_repeat, latent_dim)
noise = np.random.randn(*z_repeated.shape) * np.sqrt((alpha * variances))  # broadcasting appliqué

z_new = z_repeated + noise
z_new = torch.tensor(z_new, dtype=torch.float32).to(device)

# === CALCUL DE L'INCERTITUDE PAR MC DROPOUT ===
def get_uncertainty(z_single, T):
    preds = []
    for _ in range(T):
        with torch.no_grad():
            y_pred = denormalize(model.predict_cd_only(z_single), Cd_max, Cd_min)  # shape (1, 1)
        preds.append(y_pred.cpu().item())  # scalaire
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
        Z.append(z_new[i])
        preds.append(mean_preds[i])

Z_path = f'/home/amb/bjacques/GenNet/Results/New_shapes/new_shapes_newmodel1_alpha_{alpha}.npz'
np.savez(Z_path, latent = Z, predictions = preds)
print(len(Z))



