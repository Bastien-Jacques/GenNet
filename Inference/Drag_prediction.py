### Prediction of Cd (drag coefficient)

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from Models.GenNet_skip import AutoencoderSDF
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
model_path = "Your_Model_Path.pt" #path of the weights obtained with training

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 6
latent_dim = 128 #config this parameters with same values used during the model's training
hidden_dim = 256

model = AutoencoderSDF(latent_dim, hidden_dim, dropout=0.0) #use the same dropout value as used during training
model.load_state_dict(torch.load(model_path, map_location=device)) #loading of the model weights

model.to(device)
model.eval()

# Prepare data
val_dataset = H5SDFDataset(val_path) #validation dataset
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

gt_list = []
for batch in val_loader:
    Cd_gt = batch['Cd'].numpy().reshape(-1) 
    gt_list.append(denormalize(Cd_gt, Cd_max, Cd_min))
y_true = np.concatenate(gt_list, axis=0).reshape(-1)  # (N,) #True valued of the drag coefficient Cd 
#for each car shape of the validation set

shape_ids = val_dataset.get_ids() #(native method from the created class H5Dataset)

#partition of the ground truth value (Cd) into the 3 different car shapes
y_true_estate = []
y_true_fastback = []
y_true_notchback = []

for index in range(len(shape_ids)):
    # Ranking per car shape
    shape_id = shape_ids[index]
    if shape_id.startswith('E'):
            y_true_estate.append(y_true[index])
    elif shape_id.startswith('F'):
            y_true_fastback.append(y_true[index])
    elif shape_id.startswith('N'):
            y_true_notchback.append(y_true[index])
    else:
        print(f"Unknow ID : {shape_id}")


Cd_estate = []
Cd_notchback = []
Cd_fastback = []
Cd_full = []

start_time = time.time() #to measure inference time
with torch.no_grad():
    for batch in tqdm(val_loader):
        points = batch['points'].to(device)
        sdf = batch['sdf'].to(device)
        ids = batch['shape_id']

        
        _, Cd_pred_batch, _ = model(points, sdf)
        B = len(ids)
        for i in range(B):
            shape_id = ids[i]
            Cd_pred = denormalize(Cd_pred_batch[i].cpu().item(), Cd_max, Cd_min) #denormalisation of the predicted Cd

            # Ranking per car shape
            if shape_id.startswith('E'):
                Cd_estate.append(Cd_pred)
            elif shape_id.startswith('F'):
                Cd_fastback.append(Cd_pred)
            elif shape_id.startswith('N'):
                Cd_notchback.append(Cd_pred)
            else:
                print(f"Unknow ID : {shape_id}")
            Cd_full.append(Cd_pred)

end_time = time.time()

# Definition of severals metrics #

#Mean Squared Error
def MSE(Cd_pred, Cd_true):
    return np.mean((np.array(Cd_pred) - np.array(Cd_true))**2)
    
#Mean Relarive Absolute Error
def MRAE(Cd_pred, Cd_true):
    return np.mean(np.abs(np.array(Cd_pred) - np.array(Cd_true))/np.abs(np.array(Cd_true)))

# Mean Absolute Error
def MAE(Cd_pred, Cd_true):
    return np.mean(np.abs(np.array(Cd_pred) - np.array(Cd_true)))

# Max Absolute Error
def MaxMAE(Cd_pred, Cd_true):
     return max(np.abs(np.array(Cd_pred) - np.array(Cd_true)))

#R² score
def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

#std
def ecart_type(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    err = np.abs(y_true - y_pred)
    return np.std(err)
     

# Results

print("\n=== Results MSE Cd ===")
print(f"MSE Estateback : {MSE(Cd_estate, y_true_estate):.6f} (N={len(Cd_estate)})")
print(f"MSE Fastback   : {MSE(Cd_fastback, y_true_fastback):.6f} (N={len(Cd_fastback)})")
print(f"MSE Notchback  : {MSE(Cd_notchback, y_true_notchback):.6f} (N={len(Cd_notchback)})")
print(f"MSE Full:{MSE(Cd_full, y_true):.6f} (N={len(Cd_full)})")

print("\n=== Results MRAE Cd ===")
print(f"MRAE Estateback : {MRAE(Cd_estate, y_true_estate):.6f}, {ecart_type(Cd_estate, y_true_estate)} (N={len(Cd_estate)})")
print(f"MRAE Fastback   : {MRAE(Cd_fastback, y_true_fastback):.6f}, {ecart_type(Cd_fastback, y_true_fastback)} (N={len(Cd_fastback)})")
print(f"MRAE Notchback  : {MRAE(Cd_notchback, y_true_notchback):.6f}, {ecart_type(Cd_notchback, y_true_notchback)} (N={len(Cd_notchback)})")
print(f"MRAE Full:{MRAE(Cd_full, y_true):.6f}, {ecart_type(Cd_full, y_true)} (N={len(Cd_full)})")

print("\n=== Results MAE Cd ===")
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

print("\n=== Inference Time ===")
print(f"Total inference time : {end_time - start_time:.2f} secondes")



print(f"model_used Newmodel1, skip-connections = {True}")

