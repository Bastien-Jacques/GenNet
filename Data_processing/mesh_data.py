import os
import numpy as np
import yaml
import h5py

# Charger le config.yaml
with open("/home/amb/bjacques/GenNet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

val_path = config['data']['val_path']
with h5py.File(val_path, 'r') as f:
    shape_id = np.array(f['ids']).astype(str)

# Dossier contenant les fichiers .npz (SDF de chaque forme)
npz_folder = '/home/amb/bjacques/npz_files' 

# Liste des fichiers .npz triés
npz_files = [f for f in np.sort(os.listdir(npz_folder)) if f.endswith('.npz')]

# Chargement des ids de test set
shape_id_val = set(shape_id) 

# Dictionnaire résultat
Result = {'shape_id': [], 'vertices': [], 'faces': []}

for file in npz_files:
    filepath = os.path.join(npz_folder, file)
    data = np.load(filepath)

    # Parcours des clés du fichier npz
    keys = list(data.keys())

    # Supprimer les suffixes "_vertices" ou "_faces"
    for k in keys:
        if k.endswith("_vertices"):
            base_name = k.replace("_vertices", "")
            if base_name in shape_id_val:
                Result['shape_id'].append(base_name)
                Result['vertices'].append(data[base_name + '_vertices'])
                Result['faces'].append(data[base_name + '_faces'])


noms_vus = []
for name in Result['shape_id']:
    if name in noms_vus:
        print(name)
    noms_vus.append(name)


shape_to_remove = 'N_S_WW_WM_349'
if shape_to_remove in Result['shape_id']:
    idx = Result['shape_id'].index(shape_to_remove)

    # Supprimer aux bons indices dans toutes les listes
    for key in Result:
        del Result[key][idx]

    print(f"{shape_to_remove} supprimé de Result.")
else:
    print(f"{shape_to_remove} non trouvé dans Result.")


output_path = "/home/amb/bjacques/GenNet/data_split/val_mesh.h5"

with h5py.File(output_path, 'w') as f:
    shape_ids = [s.encode('utf-8') for s in Result['shape_id']]  # encodage en bytes
    f.create_dataset('shape_id', data=shape_ids)

    grp_vertices = f.create_group('vertices')
    grp_faces = f.create_group('faces')

    for i, shape_id in enumerate(Result['shape_id']):
        verts = Result['vertices'][i]
        facs  = Result['faces'][i]

        grp_vertices.create_dataset(shape_id, data=verts)
        grp_faces.create_dataset(shape_id, data=facs)

print("Fichier HDF5 avec maillages variables sauvegardé avec succès.")

