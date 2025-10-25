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






