## Visualisation ##

import numpy as np
import torch
from torch.utils.data import DataLoader
from Models.GenNet_skip import AutoencoderSDF 
from H5Dataset import H5SDFDataset
from skimage.measure import marching_cubes
import yaml
from tqdm import tqdm
import trimesh

# Charger le config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup
device = torch.device("cpu")

val_path  = '/GenNet/data_split/val_data.h5'
model_path = "/GenNet/Models/model.pt"

# Dataset
test_dataset = H5SDFDataset(val_path)

model = AutoencoderSDF(128, 256, 0.05).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Batch test ===
batch_size_test = 12
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
example = next(iter(test_loader))

points = example['points'].to(device)  # (B, N, 3)
sdf = example['sdf'].to(device)        # (B, N)
shape_ids = example['shape_id']
# === latent encoding ===
model.eval()
with torch.no_grad():
    z_batch = model.encoder(points, sdf) #(B,z_dim,1)

id = 4
z = z_batch[id].unsqueeze(0)
shape_id = shape_ids[id]


def reconstruct_and_export(z, shape_id):

    grid_size = 512
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z_lin = np.linspace(-1, 1, grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z_lin, indexing='ij')
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    batch_points = 128**3 
    sdf_pred_list = []

    with torch.no_grad():
        for i in range(0, grid_points.shape[0], batch_points):
            pts_batch = grid_points[i:i+batch_points]
            pts_batch_tensor = torch.from_numpy(pts_batch).float().unsqueeze(0).to(device)
            sdf_batch, _ = model.decoder(pts_batch_tensor, z)
            sdf_pred_list.append(sdf_batch.squeeze(0).cpu().numpy())

    sdf_pred = np.concatenate(sdf_pred_list, axis=0)
    sdf_3d = sdf_pred.reshape((grid_size, grid_size, grid_size))

    # Marching Cubes
    verts, faces, _, _ = marching_cubes(sdf_3d, level=0.0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Cleaning
    components = mesh.split(only_watertight=False)
    filtered = [c for c in components if c.area > 1e-4]
    mesh = max(filtered, key=lambda c: c.area)

    out_file = f'Shape_visuakisation_{shape_id}.stl'
    mesh.export(out_file)
    print(f'{out_file} done')

reconstruct_and_export(z, shape_id)

## In this code, we make the visualisation of car shaped of the validation set. 
# You can modify the code easily to generate the visualisation of new shapes using their latent vectors or directly SDF

