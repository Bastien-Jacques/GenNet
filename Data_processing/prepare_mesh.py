import os
import numpy as np
import yaml
import h5py

## This code enables the transformation of the npz files encoding meshes of the vehicules into h5 files

## We use this files during inference of the model in order to measure Chamer Distance between 
## reconstructed shapes and ground truth

# Loading config.yaml
with open("/home/amb/bjacques/GenNet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

val_path = config['data']['val_path'] # example for validation
with h5py.File(val_path, 'r') as f:
    shape_id = np.array(f['ids']).astype(str)

# folder containing npz files
npz_folder = '/home/amb/bjacques/npz_files' 

# list of sorted files
npz_files = [f for f in np.sort(os.listdir(npz_folder)) if f.endswith('.npz')]

# Loadind of ids
shape_id_val = set(shape_id) 

# Result dictionnary
Result = {'shape_id': [], 'vertices': [], 'faces': []}

for file in npz_files:
    filepath = os.path.join(npz_folder, file)
    data = np.load(filepath)
    keys = list(data.keys())

    # delete "_vertices" ou "_faces"
    for k in keys:
        if k.endswith("_vertices"):
            base_name = k.replace("_vertices", "")
            if base_name in shape_id_val:
                Result['shape_id'].append(base_name)
                Result['vertices'].append(data[base_name + '_vertices'])
                Result['faces'].append(data[base_name + '_faces'])


output_path = "YourOutputPath.h5"

with h5py.File(output_path, 'w') as f:
    shape_ids = [s.encode('utf-8') for s in Result['shape_id']]  # bytes encoding
    f.create_dataset('shape_id', data=shape_ids)

    grp_vertices = f.create_group('vertices')
    grp_faces = f.create_group('faces')

    for i, shape_id in enumerate(Result['shape_id']):
        verts = Result['vertices'][i]
        facs  = Result['faces'][i]

        grp_vertices.create_dataset(shape_id, data=verts)
        grp_faces.create_dataset(shape_id, data=facs)

print("file HDF5 with meshes saved with success.")

