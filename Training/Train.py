## Training of the model ##

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
sys.path.append("/GenNet")
from Models.GenNet_skip_v2 import AutoencoderSDF 
from H5Dataset import H5SDFDataset
import yaml


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#define hyperparameters

lr = 1e-3 #learning rate
dropout = 0.05 #dropout
lmbda = 0.05 #weight of Cd in the loss
delta = 0.1 #L1 clamped loss 
epochs = 250 #number of epochs

latent_dim = 128
hidden_dim = 512

# Initialise W&B
wandb.login(key = 'Your Wandb key')
wandb.init(project="GenNet", name='Newmodel', config={'lr':lr,
                                     'dropout':dropout,
                                     'delta':delta,
                                     'lambda':lmbda,
                                     'Loss_sdf':'L1_Clamped',
                                     'Eikonal':False,
                                     'epochs':epochs,
                                     'skip-connection': True,
                                     'hidden_layers_sdf':6,
                                     'hidden_layers_Cd':2,
                                     'latent_dim':latent_dim,
                                     'hidden_dim':hidden_dim})

train_path = config['data']['train_path']
val_path = config['data']['val_path']

train_dataset = H5SDFDataset(train_path)
val_dataset   = H5SDFDataset(val_path)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                          num_workers=8, pin_memory=True)

val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False,
                          num_workers=8, pin_memory=True)

##(adapt batch_size to your GPU's VRAM)

### -- Training -- ###

model = AutoencoderSDF(latent_dim, hidden_dim, dropout).to(device) #model definition
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optimizer  = Adam

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

best_val_loss = float('inf') #initialisation of val_loss (=infinity at first)

best_model_path = config['output']['model_path']

## L1 clamped Loss ##

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

          loss_sdf = clamped_l1_loss(sdf_pred, sdf, delta) #loss L1 clamped between sdf_pred and sdf (GT)

          loss_cd = F.mse_loss(cd_pred.squeeze(-1), Cd) #loss MSE entre Cd_pred et Cd (vérité terrain)
          loss = loss_sdf + lmbda * loss_cd 

        # gradient descent
            
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

    ## Validation ## (no backpropagation)

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
      print(f"New best modl saved: val_total_loss = {val_total_loss:.4f}, val_loss_cd = {val_loss_cd}, val_loss_sdf = {val_loss_sdf}")

    val_total_plot = val_loss_sdf + val_loss_cd
    
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

    scheduler.step(val_total_loss) 
    for param_group in optimizer.param_groups:
      print(f"Current LR: {param_group['lr']}")


