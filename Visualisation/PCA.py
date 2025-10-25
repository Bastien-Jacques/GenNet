### PCA analysis ###

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
sys.path.append("/GenNet")
from Models.GenNet import AutoencoderSDF
from Models.GenNet import SDFDecoder
from Models.GenNet import SDFEncoder
import yaml
from H5Dataset import H5SDFDataset
from tqdm import tqdm
import h5py
import json


with open('cd_stats.json', 'r') as f:
    cd_stats = json.load(f)

Cd_min = cd_stats['Cd_min']
Cd_max = cd_stats['Cd_max']

def denormalize(Cd, Cd_max, Cd_min): 
  return Cd * (Cd_max - Cd_min) + Cd_min
    

with open("/GenNet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

latent_dim = config['model']['latent_dim']
hidden_dim = config['model']['hidden_dim']
dropout = config['training']['dropout']
print(dropout)

model_path = "/GenNet/Models/model.pt"

model = AutoencoderSDF(latent_dim, hidden_dim, dropout).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

latent_vectors = []
cd_values = []
ids = []

train_path = config['data']['train_path']
batch_size = config['training']['train_batch_size']

# Load Dataset and Dataloader
train_dataset = H5SDFDataset(train_path)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


with torch.no_grad():
    for batch in tqdm(train_loader):
        points = batch['points'].to(device)
        sdf = batch['sdf'].to(device)
        Cd = denormalize(batch['Cd'].cpu().numpy(), Cd_max, Cd_min)
        id = batch['shape_id']
        ids.append(id)

        z = model.encoder(points, sdf)  # Shape: (B, latent_dim)

        latent_vectors.append(z.cpu().numpy())
        cd_values.append(Cd)

# batchs concatenation
latent_vectors = np.concatenate(latent_vectors, axis=0)
cd_values = np.concatenate(cd_values, axis=0)

# PCA 2D
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_vectors)


with h5py.File("Your_dep/latent_space.h5", 'w') as f:
    f.create_dataset("latent_2d", data=latent_2d)
    f.create_dataset("latent_vectors", data=latent_vectors)
    f.create_dataset("cd_values", data=cd_values)
    f.create_dataset('shape_id', data = ids)



