import os
import numpy as np
import h5py
from tqdm import tqdm

def merge_npz_to_h5(npz_dir, h5_output_path):
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])

    if len(npz_files) == 0:
        raise RuntimeError(f"Aucun fichier .npz trouvé dans {npz_dir}")

    # Préparation des conteneurs
    all_points = []
    all_sdf = []
    all_cd = []
    all_ids = []

    print(f"[INFO] Lecture des {len(npz_files)} fichiers .npz...")

    for file in tqdm(npz_files):
        path = os.path.join(npz_dir, file)
        data = np.load(path)

        points = data['points'].astype(np.float32)   # (P, 3)
        sdf    = data['sdf'].astype(np.float32)      # (P,)
        cd     = data['cd'].astype(np.float32) if 'cd' in data else np.array(0.0, dtype=np.float32)

        if points.shape[0] != sdf.shape[0]:
            print(f"[WARNING] {file}: mismatch entre points et sdf → ignoré.")
            continue

        all_points.append(points)
        all_sdf.append(sdf)
        all_cd.append(cd)
        all_ids.append(file.replace('.npz', '').encode('utf-8'))

    # Longueur du dataset
    N = len(all_points)

    if N == 0:
        raise RuntimeError("Aucune donnée valide trouvée.")

    print(f"[INFO] Création du fichier HDF5 : {h5_output_path}")

    with h5py.File(h5_output_path, 'w') as f:
        # Dimensions : (N, P, 3) et (N, P)
        P = all_points[0].shape[0]
        f.create_dataset('points', data=np.stack(all_points))         # shape: (N, P, 3)
        f.create_dataset('sdf', data=np.stack(all_sdf))               # shape: (N, P)
        f.create_dataset('cd', data=np.array(all_cd))                 # shape: (N,)
        f.create_dataset('ids', data=np.array(all_ids))              # shape: (N,)

    print(f"[OK] Fusion terminée. {N} entrées sauvegardées dans {h5_output_path}")

# === Utilisation ===
npz_folder = '/GenNet/SDF_ModelNet40'
output_h5  = '/GenNet/data_split/ModelNet40.h5'

merge_npz_to_h5(npz_folder, output_h5)
