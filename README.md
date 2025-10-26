# GenNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

GenNet is a deep learning framework for predicting the aerodynamic drag coefficient of vehicle shapes and reconstructing their Signed Distance Functions (SDF). This enables the generation of new car geometries optimized for aerodynamic performance. GenNet is based on an autoencoder structure. The model has been trained on the DrivAerNet++ Dataset (train set). This repository contains the source code, analysis notebooks, and scripts to train and evaluate the model. You can also read the full work by [clicking on the link](./Mémoire.pdf), where all details concerning the mechanics part and the data processing are given.

## 📋 Table of Contents
- [Installation](#-installation)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🛠 Installation

### Prerequisites
- Python 3.9+
- pip (or conda)

### Clone the repository
```bash
git clone https://github.com/Bastien-Jacques/GenNet.git
cd GenNet
```
### Create a virtual environement
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```
## 🚀 Usage

### 🧠 Train the Model
To train GenNet on the DrivAerNet++ dataset using the provided configuration file:
```python
train.py --config config.yaml
```
This will start the training process and save model checkpoints in the checkpoints/ directory.

### 🧩 Inference Scripts

The `Inference/` folder contains several scripts for evaluating and post-processing the trained GenNet model:

| Script | Description |
|--------|--------------|
| `Chamfer_Distance.py` | Computes the Chamfer Distance between predicted and ground truth geometries. |
| `Drag_prediction.py` | Predicts the aerodynamic drag coefficient (C<sub>d</sub>) for new geometries using a trained model. |
| `Drag_Optimisation.py` | Performs gradient-based optimization in the latent space to minimize drag. |
| `Drag_Random_Optimisation.py` | Performs random (non-gradient) search in the latent space to explore aerodynamic improvements. |
| `Morphing.py` | Generates morphing between two vehicle geometries by interpolation of their respective latent vectors. |
| `Uncertainty.py` | Estimates model uncertainty on drag prediction using Monte Carlo Dropout. |

Each script can be executed independently:
```python
Inference/Drag_prediction.py --weights checkpoints/best_model.pt --input data
```
by replacing [Drag_prediction] by the wanted code from the Inference folder.


## 📁 Repository Structure
```bash

GenNet/
├── Data_processing/              
│   ├── H5Dataset.py
│   ├── npz_to_h5.py
│   ├── prepare_data.py
│   └── prepare_mesh.py
|
├── Inference/              # Inference and post-processing scripts
│   ├── Chamfer_Distance.py
│   ├── Drag_Optimisation.py
│   ├── Drag_Random_Optimisation.py
│   ├── Drag_prediction.py
│   ├── Morphing.py
│   └── Uncertainty.py
│
├── Models/                 # Autoencoder architecture with or without skip-connections
│   ├── GenNet.py
│   ├── GenNet_skip.py
│   └── GenNet_skip_v2.py
|
├── Notebooks/              # Jupyter notebooks for analysis of the results
│   └── Analyse.ipynb
│   
├── Training/               # Training with or wothout Eikonal Loss 
│   ├── Train.py
│   └── Train_Eikonal.py
│
├── Visualisation/          # 3D visualizations and PCA of latent space
|   ├── Visualisation.py
│   └── PCA.py
│
├── config.yaml             # Configuration file for training
├── requirements.txt        # Python dependencies
├── cd_stats.json           # Dataset statistics
├── Mémoire.pdf             # Full research report
├── LICENSE                 # MIT license
└── README.md               # Project documentation
```


## 📊 Results
GenNet achieves accurate prediction of aerodynamic drag coefficients and high-fidelity reconstruction of 3D geometries on the **DrivAerNet++** dataset.

### Data representation
GenNet is based on a signed distance function's representation of each car sample in the **DrivAerNet++** dataset. Signed Distance Function (SDF) is defined as the function:

$$
\text{SDF}(\mathbf{x}) =
\begin{cases}
+d(\mathbf{x}, \partial \Omega), & \text{if } \mathbf{x} \notin \Omega \\
-d(\mathbf{x}, \partial \Omega), & \text{if } \mathbf{x} \in \Omega
\end{cases}
$$

where $d(\mathbf{x}, \partial \Omega) = \min_{\mathbf{p} \in \partial \Omega} \| \mathbf{x} - \mathbf{p} \|_2$ is the Euclidean distance from a point $\mathbf{x}$ to the surface boundary $\partial \Omega$.

The geometries of the different vehicles in the DrivAerNet++ dataset are originally provided as meshes. The conversion to SDF format was performed by sampling 250,000 points per vehicle within the cube [-1,1]³, after normalization and centering of the meshes.
The isotropic normalization ensures that the model learns patterns related to the shape of the vehicles rather than their absolute size.

To avoid excessive computational cost while maintaining a faithful representation of each vehicle’s structure, 80% of the 250,000 points were sampled close to the surface by adding Gaussian noise (standard deviation 0.01, zero mean) relative to the surface.
The remaining 20% of the points were sampled randomly within the cube.

<p align="center">
  <img src="docs/échantillonnage-SDF_250k_pts.JPG" width="55%">
  <img src="docs/SDF_250k_epsilon.jpg" width="30%">
</p>

<p align="center">
  <b>Left:</b> The red points correspond to locations outside the mesh (SDF > 0), while the blue points are inside the mesh (SDF < 0).  <b>Right:</b> Distribution of sampling points as a function of the distance to the mesh
</p>

The geometry of a vehicle is thus fully represented implicitly through the 0-isosurface of the Signed Distance Function. The conversion from meshes to SDF enables an accurate representation of vehicle geometries, as well as the use of fully connected networks and inverse shape generation.

The drag coefficients were normalized using the Min–Max scaling method, with the minimum and maximum values computed only on the training set to prevent any data leakage between the different subsets.
This normalization helps stabilize the training process by ensuring proper gradient propagation.

### Model 
















