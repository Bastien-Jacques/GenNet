import torch
from torch.utils.data import Dataset
import h5py

### This code enables the creation of torch tensors using .h5 files ###

## data loading ##

class H5SDFDataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        self.h5_file = None  # Lazy loading

        # On lit la taille sans charger les données
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['cd'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Lazy open pour chaque worker
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        points = torch.tensor(self.h5_file['points'][idx], dtype=torch.float32)  # (P, 3)
        sdf    = torch.tensor(self.h5_file['sdf'][idx], dtype=torch.float32)     # (P,)
        cd     = torch.tensor(self.h5_file['cd'][idx], dtype=torch.float32)      # ()
        shape_id = self.h5_file['ids'][idx].decode('utf-8')

        return {
            'points': points,
            'sdf': sdf,
            'Cd': cd,
            'shape_id': shape_id
        }
    
    def get_ids(self):
        
        with h5py.File(self.h5_path, 'r') as f:
            ids = [id_str.decode('utf-8') for id_str in f['ids']]
        return ids
        
    def get_Cd(self):
        with h5py.File(self.h5_path, 'r') as f:
            Cd = [Cd for Cd in f['Cd']]
        return Cd
