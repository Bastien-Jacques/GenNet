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
with open("/home/amb/bjacques/GenNet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_path  = config['data']['test_path']
model_path = "/home/amb/bjacques/GenNet/Weights/best_model_test5.pt"

# Dataset et DataLoader
test_dataset = H5SDFDataset(test_path)
batch_size_test = 12
test_loader  = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=8, pin_memory=True)

# Charger le modèle
model = AutoencoderSDF(128, 256, 0).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# IDs ciblés
id_Fastback  = 'F_D_WM_WW_3215'
id_Notchback = 'N_S_WWS_WM_331'
target_ids = [id_Fastback, id_Notchback]

Result = {'shape_id':[],
          'latent':[]}

# Boucle DataLoader
for batch in tqdm(test_loader, desc="Processing test set"):
    points   = batch['points'].to(device)
    sdf      = batch['sdf'].to(device)
    shape_id = batch['shape_id']  # reste sur CPU (liste de strings)

    with torch.no_grad():
        z_batch = model.encoder(points, sdf)

    for idx in range(points.shape[0]):
        sid = shape_id[idx]
        if sid not in target_ids:
            continue  # on saute les échantillons qui ne sont pas ciblés

        z = z_batch[idx].unsqueeze(0)
        Result['shape_id'].append(sid)
        Result['latent'].append(z)

def reconstruct_and_export(z, alpha):

    grid_size = 512
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z_lin = np.linspace(-1, 1, grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z_lin, indexing='ij')
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    batch_points = 128**3  # nombre de points traités par batch
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

    # Nettoyage
    #components = mesh.split(only_watertight=False)
    #filtered = [c for c in components if c.area > 1e-4]
    #if not filtered:
        #print(f"Aucun composant significatif pour {shape_id}")
        #return
    #mesh = max(filtered, key=lambda c: c.area)

    out_file = f'/home/amb/bjacques/GenNet/Results/mesh_reconstruction/test5_mesh_alpha_{alpha:.1f}.stl'
    mesh.export(out_file)
    print(f'{out_file} done')

alpha_range = np.arange(0,1.1,0.1)

for alpha in alpha_range:
    z = alpha * Result['latent'][0] + (1-alpha)*Result['latent'][1]
    reconstruct_and_export(z,alpha)
