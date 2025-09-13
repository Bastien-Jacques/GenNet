import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import sys
sys.path.append("/home/amb/bjacques/GenNet")
from Models.GenNet_skip_v2 import AutoencoderSDF 
from H5Dataset import H5SDFDataset
import yaml

# Charger le fichier YAML des Hyperparamètres

with open("/home/amb/bjacques/GenNet/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

lr = 1e-3
dropout = 0.05
lmbda = 0.05
delta = 0.1
epochs = 250

latent_dim = 128
hidden_dim = 512




# Initialise W&B
wandb.login(key = '033fda8a1b7e6d603097e8689d269a7b52814c9a')
wandb.init(project="GenNet_New", name='Newmodel_3', config={'lr':lr,
                                     'dropout':dropout,
                                     'delta':delta,
                                     'lambda':lmbda,
                                     'Loss_sdf':'L1_Clamped',
                                     'Eikonal':False,
                                     'epochs':epochs,
                                     'skip-connection': True,
                                     'hidden_layers_sdf':6,
                                     'hidden_layers_sc':2,
                                     'latent_dim':latent_dim,
                                     'hidden_dim':hidden_dim})
## Création des batchs pour entraînement ##

train_path = config['data']['train_path']
val_path = config['data']['val_path']

train_dataset = H5SDFDataset(train_path)
val_dataset   = H5SDFDataset(val_path)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          num_workers=8, pin_memory=True)

val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False,
                          num_workers=8, pin_memory=True)


### -- Entraînement du modèle -- ###

model = AutoencoderSDF(latent_dim, hidden_dim, dropout).to(device) #définition du modèle
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optimizer  = Adam

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=wandb.config.step_size, gamma=wandb.config.gamma)
#lr scheduler réduit toutes les patience épochs sans prendre en compte l'évolutio de la val loss

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1, 
    patience=10, 
    threshold=1e-4, 
    threshold_mode='rel', 
    cooldown=0, 
    min_lr=1e-6, 
    eps=1e-08)


#création d'un scheduler pour affiner lr, configuré avec wandb

best_val_loss = float('inf') #initialisation de la meilleure val_loss = infinie au début

best_model_path = "/home/amb/bjacques/GenNet/Weights/Newmodel_3.pt"
#path de sauvegarde du meilleur modèle (selon la val loss) dans le dossier Weights

## définition de la L1 clamped Loss ##

def clamped_l1_loss(pred, target, delta):
    pred_clamped = torch.clamp(pred, min=-delta, max=delta)
    target_clamped = torch.clamp(target, min=-delta, max=delta)
    return torch.abs(pred_clamped - target_clamped).mean()


## Training ##


for epoch in range(1, epochs + 1):

    model.train()
    train_loss_sdf = 0.0
    train_loss_cd  = 0.0
    num_batches = 0

    with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
        for batch in pbar:

          points = batch['points'].to(device)
          sdf = batch['sdf'].to(device)
          Cd = batch['Cd'].to(device)

          sdf_pred, cd_pred, _ = model(points, sdf)

          loss_sdf = clamped_l1_loss(sdf_pred, sdf, delta) #loss L1 clamped entre sdf_pred et sdf (vérité terrain)
          #loss_sdf = F.mse_loss(sdf_pred, sdf) #loss MSE entre sdf_pred et sdf (vérité terrain)

          #rajouter loss eikonal dans la suite

          loss_cd = F.mse_loss(cd_pred.squeeze(-1), Cd) #loss MSE entre Cd_pred et Cd (vérité terrain)
          loss = loss_sdf + lmbda * loss_cd #lambda à paramétrer

        # descente de gradient

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          train_loss_sdf += loss_sdf.item()
          train_loss_cd  += loss_cd.item()
          num_batches += 1

          pbar.set_postfix({
                "train_sdf": loss_sdf.item(),
                "train_Cd":  loss_cd.item()
            })

    train_loss_sdf /= num_batches
    train_loss_cd  /= num_batches
    train_total_loss = train_loss_sdf + lmbda * train_loss_cd

    ## Validation ## (pas de backpropagation)

    model.eval()
    val_loss_sdf = 0.0
    val_loss_cd = 0.0
    num_batches_val = 0

    with torch.no_grad():
        for batch in val_loader:
            points = batch['points'].to(device)
            sdf = batch['sdf'].to(device)
            Cd = batch['Cd'].to(device)

            sdf_pred, cd_pred, _ = model(points, sdf)

            loss_sdf = clamped_l1_loss(sdf_pred, sdf, delta)
            loss_cd = F.mse_loss(cd_pred.squeeze(-1), Cd)

            val_loss_sdf += loss_sdf.item()
            val_loss_cd  += loss_cd.item()
            num_batches_val += 1

    val_loss_sdf /= num_batches_val
    val_loss_cd  /= num_batches_val
    val_total_loss = val_loss_sdf + lmbda * val_loss_cd

    if val_total_loss < best_val_loss:
      best_val_loss = val_total_loss
      torch.save(model.state_dict(), best_model_path)
      print(f"Nouveau meilleur modèle sauvegardé val_total_loss = {val_total_loss:.4f}, val_loss_cd = {val_loss_cd}, val_loss_sdf = {val_loss_sdf}")

      # Crée un artifact pour stocker ce modèle dans wandb
      artifact = wandb.Artifact("best_model", type="model")
      artifact.add_file(best_model_path)
      wandb.log_artifact(artifact)

    val_total_plot = val_loss_sdf + val_loss_cd
    # pas de lambda pour pouvoir comparer entre eux les différents entrainements avec des lambda différents

    wandb.log({
    "epoch": epoch,
    "train_loss_sdf": train_loss_sdf,
    "train_loss_Cd": train_loss_cd,
    "train_total_loss": train_total_loss,
    "val_loss_sdf": val_loss_sdf,
    "val_loss_Cd": val_loss_cd,
    "val_total_loss": val_total_loss,
    "learning_rate": optimizer.param_groups[0]['lr']
    })

    scheduler.step(val_total_loss) #pour affiner ou non lr si stagnation de la val loss
    for param_group in optimizer.param_groups:
      print(f"Current LR: {param_group['lr']}")


