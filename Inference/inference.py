
### Inférence du modèle ###

# Objectif: Calculer l'incertitude epistémique sur les prédictions de Cd #
#utilisation du MC Dropout

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.GenNet_skip_v2 import AutoencoderSDF 
from H5Dataset import H5SDFDataset
import yaml
from tqdm import tqdm
import json

# Charger le config.yaml
with open("/home/amb/bjacques/GenNet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

val_path = config['data']['val_path']
batch_size = 12


model_path = "/home/amb/bjacques/GenNet/Weights/Newmodel_2.pt"

# Charger dataset et dataloader
val_dataset = H5SDFDataset(val_path)
shape_ids = val_dataset.get_ids()
val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

latent_dim = 128
hidden_dim = 256

dropout = 0.05

# Charger modèle
model = AutoencoderSDF(latent_dim, hidden_dim, dropout).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


with open('/home/amb/bjacques/GenNet/data_split/cd_stats.json', 'r') as f:
    cd_stats = json.load(f)

Cd_min = cd_stats['Cd_min']
Cd_max = cd_stats['Cd_max']

def denormalize(Cd, Cd_max, Cd_min): #dénormalise une valeur Cd
  return Cd * (Cd_max - Cd_min) + Cd_min

def enable_dropout(model): #Active le dropout en mode inference pour MC Dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

model.eval()
enable_dropout(model)
MC = 100  # nombre de passes Monte Carlo

Result = {'Cd_pred':[]}

with torch.no_grad():
    for i in tqdm(range(MC)):
        all_preds = []
        for batch in val_loader:
            points = batch['points'].to(device)
            sdf = batch['sdf'].to(device)

            # Forward avec dropout actif (en eval, si MC Dropout est bien configuré)
            _, Cd_pred, _ = model(points, sdf)

            # Dénormalise et stocke la prédiction
            Cd_pred = denormalize(Cd_pred.cpu().numpy(), Cd_max, Cd_min)
            all_preds.append(Cd_pred)
        Result['Cd_pred'].append(np.concatenate(all_preds, axis=0))

# Convertir en (MC, N_samples)
predictions = np.stack(Result['Cd_pred'], axis=0)

mean_preds = predictions.mean(axis=0)
std_preds = predictions.std(axis=0)  # Epistemic uncertainty

mu = mean_preds.reshape(-1)          # (N,)
sigma_mc = std_preds.reshape(-1)     # (N,)

mean_std = np.mean(sigma_mc)

print(f'moyenne des incertitudes non-calibrées: {mean_std}')
print(mu, f'moyenne des prédictions sur les {MC} passes')
print(sigma_mc, 'écart type sur les prédictions' )
print(f'dropout utilisé: {dropout}')


file_path = '/home/amb/bjacques/GenNet/Results/MC_dropout_val.npz'
np.savez(file_path,
         incertitudes=sigma_mc,
         Moyennes=mu,
         moyenne_incertitudes=mean_std,
         dropout=dropout,
         shape_id = shape_ids,
         predictions = predictions,
         N_passes = MC)
print('fichier sauvegardé')