import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import h5py
from tqdm import tqdm
import trimesh
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree
from Models.GenNet_skip import AutoencoderSDF
from H5Dataset import H5SDFDataset
import pickle


def normalize_and_center(mesh):
    mesh.apply_translation(-mesh.centroid)
    scale = np.max(mesh.bounding_box.extents) / 2.0
    mesh.apply_scale(1.0 / scale)
    return mesh

def chamfer_distance(p1, p2):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    dist1, _ = tree1.query(p2)
    dist2, _ = tree2.query(p1)
    return np.mean(dist1**2) + np.mean(dist2**2)

def mesh_cleaned(mesh, alpha):
    # Split into connected components
    components = mesh.split(only_watertight=False)

    # Filter out small ones (based on surface area or number of faces)
    filtered = [c for c in components if c.area > alpha]

    # Optionally: keep only the largest
    mesh = max(filtered, key=lambda c: c.area)

    return mesh


# Load config
with open("/home/amb/bjacques/GenNet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

val_path = config['data']['val_path']
model_path = "/home/amb/bjacques/GenNet/Weights/best_model_test5.pt"

gt_meshes_path = "/home/amb/bjacques/GenNet/data_split/val_gt_meshes.pkl"

with open(gt_meshes_path, 'rb') as f:
    gt_meshes = pickle.load(f)



# Setup
device = torch.device("cpu")
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

grid_size = 64
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
z_lin = np.linspace(-1, 1, grid_size)
grid_x, grid_y, grid_z = np.meshgrid(x, y, z_lin, indexing='ij')
grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)  # (N, 3)
grid_points_tensor = torch.from_numpy(grid_points).float().to(device)  # (N, 3)

# Inference + Chamfer par catégorie
chamfer_estate = []
chamfer_fastback = []
chamfer_notchback = []
chamfer_full = []

with torch.no_grad():
    for batch in tqdm(val_loader):
        points = batch['points'].to(device)
        sdf = batch['sdf'].to(device)
        ids = batch['shape_id']
        
        z_batch = model.encoder(points, sdf)  # (B, latent_dim)
        B = z_batch.shape[0]

        grid_points_expanded = grid_points_tensor.unsqueeze(0).repeat(B, 1, 1)  # (B, N, 3)

        sdf_pred_batch, _ = model.decoder(grid_points_expanded, z_batch)  # (B, N)
        sdf_pred_batch = sdf_pred_batch.cpu().numpy()

        for i in range(B):
            shape_id = ids[i]
            sdf_pred = sdf_pred_batch[i].reshape((grid_size, grid_size, grid_size))
            
            verts, faces, _, _ = marching_cubes(sdf_pred, level=0.0, spacing=(2/grid_size,)*3)
            verts -= 1.0

            mesh_pred = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh_pred = normalize_and_center(mesh_pred)
            #mesh_pred = mesh_cleaned(mesh_pred, 1e-4)

            points_pred = mesh_pred.sample(5000)
            points_gt = gt_meshes[shape_id].sample(5000)

            chamfer = chamfer_distance(points_pred, points_gt)

            # Classement par type de voiture
            if shape_id.startswith('E'):
                chamfer_estate.append(chamfer)
            elif shape_id.startswith('F'):
                chamfer_fastback.append(chamfer)
            elif shape_id.startswith('N'):
                chamfer_notchback.append(chamfer)
            else:
                print(f"ID inconnu : {shape_id}")
            chamfer_full.append(chamfer)

# Résultats
print("\n=== Résultats Chamfer Distance ===")
print(f"Estateback : {np.mean(chamfer_estate):.6f} {np.std(chamfer_estate):.6f} (N={len(chamfer_estate)})")
print(f"Fastback   : {np.mean(chamfer_fastback):.6f} {np.std(chamfer_fastback):.6f} (N={len(chamfer_fastback)})")
print(f"Notchback  : {np.mean(chamfer_notchback):.6f} {np.std(chamfer_notchback):.6f} (N={len(chamfer_notchback)})")
print(f"Full:{np.mean(chamfer_full):.6f} {np.std(chamfer_full):.6f} (N = {len(chamfer_full)})")

print(f"model_used GenNet test5, skip-connections = {True}")

