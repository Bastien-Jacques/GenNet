### Predictions du Cd

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from Models.GenNet_skip_v2 import AutoencoderSDF
from H5Dataset import H5SDFDataset
import json

with open('/home/amb/bjacques/GenNet/data_split/cd_stats.json', 'r') as f:
    cd_stats = json.load(f)

Cd_min = cd_stats['Cd_min']
Cd_max = cd_stats['Cd_max']

def denormalize(Cd, Cd_max, Cd_min): #dénormalise une valeur Cd
  return Cd * (Cd_max - Cd_min) + Cd_min

# Load config
with open("/home/amb/bjacques/GenNet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

val_path = config['data']['val_path']
model_path = "/home/amb/bjacques/GenNet/Weights/Newmodel_1.pt"

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 6
latent_dim = 128
hidden_dim = 256

model = AutoencoderSDF(latent_dim, hidden_dim, dropout=0.0)
model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval()

# Prepare data
val_dataset = H5SDFDataset(val_path)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

gt_list = []
for batch in val_loader:
    Cd_gt = batch['Cd'].numpy().reshape(-1) 
    gt_list.append(denormalize(Cd_gt, Cd_max, Cd_min))
y_true = np.concatenate(gt_list, axis=0).reshape(-1)  # (N,)

shape_ids = val_dataset.get_ids()

y_true_estate = []
y_true_fastback = []
y_true_notchback = []

for index in range(len(shape_ids)):
    # Classement par type de voiture
    shape_id = shape_ids[index]
    if shape_id.startswith('E'):
            y_true_estate.append(y_true[index])
    elif shape_id.startswith('F'):
            y_true_fastback.append(y_true[index])
    elif shape_id.startswith('N'):
            y_true_notchback.append(y_true[index])
    else:
        print(f"ID inconnu : {shape_id}")


Cd_estate = []
Cd_notchback = []
Cd_fastback = []
Cd_full = []

start_time = time.time()
with torch.no_grad():
    for batch in tqdm(val_loader):
        points = batch['points'].to(device)
        sdf = batch['sdf'].to(device)
        ids = batch['shape_id']

        
        _, Cd_pred_batch, _ = model(points, sdf)
        B = len(ids)
        for i in range(B):
            shape_id = ids[i]
            Cd_pred = denormalize(Cd_pred_batch[i].cpu().item(), Cd_max, Cd_min)

            # Classement par type de voiture
            if shape_id.startswith('E'):
                Cd_estate.append(Cd_pred)
            elif shape_id.startswith('F'):
                Cd_fastback.append(Cd_pred)
            elif shape_id.startswith('N'):
                Cd_notchback.append(Cd_pred)
            else:
                print(f"ID inconnu : {shape_id}")
            Cd_full.append(Cd_pred)

end_time = time.time()

def MSE(Cd_pred, Cd_true):
    return np.mean((np.array(Cd_pred) - np.array(Cd_true))**2)

def MRAE(Cd_pred, Cd_true):
    return np.mean(np.abs(np.array(Cd_pred) - np.array(Cd_true))/np.abs(np.array(Cd_true)))

def MAE(Cd_pred, Cd_true):
    return np.mean(np.abs(np.array(Cd_pred) - np.array(Cd_true)))


def MaxMAE(Cd_pred, Cd_true):
     return max(np.abs(np.array(Cd_pred) - np.array(Cd_true)))


def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

def ecart_type(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    err = np.abs(y_true - y_pred)
    return np.std(err)
     

# Résultats

print("\n=== Résultats MSE Cd ===")
print(f"MSE Estateback : {MSE(Cd_estate, y_true_estate):.6f} (N={len(Cd_estate)})")
print(f"MSE Fastback   : {MSE(Cd_fastback, y_true_fastback):.6f} (N={len(Cd_fastback)})")
print(f"MSE Notchback  : {MSE(Cd_notchback, y_true_notchback):.6f} (N={len(Cd_notchback)})")
print(f"MSE Full:{MSE(Cd_full, y_true):.6f} (N={len(Cd_full)})")

np.savez('/home/amb/bjacques/GenNet/Results/results_newmodel1.npz', preds = Cd_full, GT = y_true )
print('fichier sauvegardé')

print("\n=== Résultats MRAE Cd ===")
print(f"MRAE Estateback : {MRAE(Cd_estate, y_true_estate):.6f}, {ecart_type(Cd_estate, y_true_estate)} (N={len(Cd_estate)})")
print(f"MRAE Fastback   : {MRAE(Cd_fastback, y_true_fastback):.6f}, {ecart_type(Cd_fastback, y_true_fastback)} (N={len(Cd_fastback)})")
print(f"MRAE Notchback  : {MRAE(Cd_notchback, y_true_notchback):.6f}, {ecart_type(Cd_notchback, y_true_notchback)} (N={len(Cd_notchback)})")
print(f"MRAE Full:{MRAE(Cd_full, y_true):.6f}, {ecart_type(Cd_full, y_true)} (N={len(Cd_full)})")

print("\n=== Résultats MAE Cd ===")
print(f"MAE Estateback : {MAE(Cd_estate, y_true_estate):.6f} (N={len(Cd_estate)})")
print(f"MAE Fastback   : {MAE(Cd_fastback, y_true_fastback):.6f} (N={len(Cd_fastback)})")
print(f"MAE Notchback  : {MAE(Cd_notchback, y_true_notchback):.6f} (N={len(Cd_notchback)})")
print(f"MAE Full:{MAE(Cd_full, y_true):.6f} (N={len(Cd_full)})")

print("\n=== R2 Scores ===")
print(f"R2 Estateback : {r2_score(y_true_estate, Cd_estate):.4f}")
print(f"R2 Fastback   : {r2_score(y_true_fastback, Cd_fastback):.4f}")
print(f"R2 Notchback  : {r2_score(y_true_notchback, Cd_notchback):.4f}")
print(f"R2 Full       : {r2_score(y_true, Cd_full):.4f}")


print("\n=== Max MAE ===")
print(f'max MAE :{MaxMAE(Cd_full, y_true):.6f}')


print("\n=== Temps d'inférence ===")
print(f"Temps total d'inférence : {end_time - start_time:.2f} secondes")



print(f"model_used Newmodel1, skip-connections = {True}")

