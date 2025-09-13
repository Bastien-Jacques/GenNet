## This code enables the creation of the database by taking the Drag and SDF files. 
# It also makes the partionning of the data into a train set, validation set and test set

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import zipfile
import json
import gc  # pour libérer la mémoire après chaque fichier
import h5py



## structure of data##

"""
├────Drag/
│   ├── Cd_E_S_WW_WM.txt ────────────────────────|
│   ├── Cd_N_S_WWC_WM.txt                        |
│   └── ... (folders for each car shape)         |───────── E_S_WW_WM_001.npz Cd_1 each drag coefficient
│                                                           E_S_WW_WM_002.npz Cd_2
└───SDF_files/                                                         .
        ├── SDF_E_S_WW_WM_000.npz                                      .
        ├── SDF_E_S_WW_WM_001.npz                                      .
        ├── ...                                             E_S_WW_WM_700.npz Cd_700
        └── SDF_N_S_WWC_WM_680.npz
        every SDF file (8000 files)
"""

### -- Dataset creation -- ###


class SDFDataset(Dataset):
    def __init__(self, sdf_dir, cd_txt_dir):
        self.filepaths = sorted(glob.glob(os.path.join(sdf_dir, "*.npz")))

        # Lecture des .txt files to find Cd
        self.cd_dict = {}
        txt_files = sorted(glob.glob(os.path.join(cd_txt_dir, "*.txt")))
        for txt in txt_files:
            with open(txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        name = parts[0].replace('.vtk', '').replace('.npz', '')
                        try:
                            cd = float(parts[1])
                            self.cd_dict[name] = cd
                        except ValueError:
                            print(f"[WARN] Cd invalide for {name} in {txt}")

        self.valid_filepaths = []
        self.valid_cd_values = []
        self.valid_names = []

        print(f"Association of SDF and Cd ({len(self.filepaths)} files find)...")
        for path in tqdm(self.filepaths, desc="Matching SDF ↔ Cd"):
            base_name = os.path.splitext(os.path.basename(path))[0]

            # Check correspondance + valide file
            if base_name in self.cd_dict:
                try:
                    with np.load(path) as data:
                        _ = data['points']
                        _ = data['sdf']
                    self.valid_filepaths.append(path)
                    self.valid_cd_values.append(self.cd_dict[base_name])
                    self.valid_names.append(base_name)
                except (zipfile.BadZipFile, KeyError, OSError) as e:
                    tqdm.write(f"[WARN] file not valid or corrompted : {path} ({type(e).__name__})")
            else:
                tqdm.write(f"[WARN] No Cd for {base_name} → ignored.")

        assert len(self.valid_filepaths) > 0, "No SDF file associated to a Cd file was find."

    def __len__(self):
        return len(self.valid_filepaths)

    def __getitem__(self, idx):
        file = self.valid_filepaths[idx]
        try:
            with np.load(file) as data:
                points = data['points'].astype(np.float32)
                sdf = data['sdf'].astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Impossible to read {file} in __getitem__ : {e}")

        cd = self.valid_cd_values[idx]
        name = self.valid_names[idx]

        return {
            "points": torch.from_numpy(points),
            "sdf": torch.from_numpy(sdf),
            "Cd": torch.tensor(cd, dtype=torch.float32),
            "shape_id": name
        }


SDF_path = 'YOurSDFPath'
Drag_path = 'YourDragPath'
full_dataset = SDFDataset(SDF_path, Drag_path)
total_size = len(full_dataset)


# Proportions
train_ratio = 0.8
val_ratio = 0.1

# Calcul des shapes
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  

# Random Split (reproductible of fixed seed)
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)


# Normalisation of Cd after splitting the data
# Calcul of Cd_max and Cd_min on the dataset of training only ## Very important in order to prevent data leakage
train_cd_values = [item['Cd'].item() for item in train_dataset]
Cd_max = max(train_cd_values)
Cd_min = min(train_cd_values)

def export_subset_to_hdf5(dataset_subset, full_dataset, filename, normalize_cd=False, cd_min=None, cd_max=None):
    N = len(dataset_subset)
    P = 250000 

    with h5py.File(filename, 'w') as f:
        points_ds = f.create_dataset('points', shape=(N, P, 3), dtype='float32')
        sdf_ds    = f.create_dataset('sdf', shape=(N, P), dtype='float32')
        cd_ds     = f.create_dataset('cd', shape=(N,), dtype='float32')
        ids_ds    = f.create_dataset('ids', shape=(N,), dtype=h5py.string_dtype())

        for i in tqdm(range(N), desc=f"Exporting to {os.path.basename(filename)}"):
            idx = dataset_subset.indices[i]
            try:
                item = full_dataset[idx]
            except Exception as e:
                print(f"[SKIP] File {full_dataset.valid_filepaths[idx]} → {e}")
                continue

            points = item['points'].numpy()
            sdf = item['sdf'].numpy()
            cd = item['Cd'].item()
            if normalize_cd:
                cd = (cd - cd_min) / (cd_max - cd_min)

            if points.shape != (P, 3) or sdf.shape != (P,):
                print(f"[SKIP] not attended shape for {item['shape_id']}, shape points = {points.shape}")
                continue

            points_ds[i] = points
            sdf_ds[i]    = sdf
            cd_ds[i]     = cd
            ids_ds[i]    = item['shape_id']

            del item, points, sdf
            gc.collect()


#  Output directory
output_dir = 'YourOtput/dir' #(YourEnvironment/data_split) for instance
os.makedirs(output_dir, exist_ok=True)

# Save of Cd min and Cd_max for future denormalisation
cd_stats = {'Cd_min': Cd_min, 'Cd_max': Cd_max}
with open(os.path.join(output_dir, 'cd_stats.json'), 'w') as f:
    json.dump(cd_stats, f, indent=4)

export_subset_to_hdf5(train_dataset, full_dataset, os.path.join(output_dir, 'train_data.h5'),
                      normalize_cd=True, cd_min=Cd_min, cd_max=Cd_max)

export_subset_to_hdf5(val_dataset, full_dataset, os.path.join(output_dir, 'val_data.h5'),
                      normalize_cd=True, cd_min=Cd_min, cd_max=Cd_max)

export_subset_to_hdf5(test_dataset, full_dataset, os.path.join(output_dir, 'test_data.h5'),
                      normalize_cd=False)

