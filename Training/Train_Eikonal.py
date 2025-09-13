## Training of the model with Eikonal Regularization in the loss ##

import sys
sys.path.append("/home/amb/bjacques/GenNet")
from Models.GenNet_skip import AutoencoderSDF
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from H5Dataset import H5SDFDataset
import yaml


# Charger le fichier YAML des Hyperparamètres

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

lr = 1e-3
dropout = 0.05
lmbda = 0.05
delta = 0.1
epochs = 250
lambda_eikonal = 0.1

latent_dim = 128
hidden_dim = 512

# Initialise W&B
wandb.init(project="GenNet_New", name='Newmodel', config={'lr':lr,
                                     'dropout':dropout,
                                     'delta':delta,
                                     'lambda':lmbda,
                                     'lambda_eikonal': lambda_eikonal,
                                     'Loss_sdf':'L1_Clamped',
                                     'Eikonal':True,
                                     'epochs':epochs,
                                     'skip-connection': True,
                                     'hidden_layers_sdf':6,
                                     'hidden_layers_cd':2,
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


# (adapt batch_size to your GPU's VRAM)


### -- Model's training -- ###

model = AutoencoderSDF(latent_dim, hidden_dim, dropout).to(device) #model definition
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optimizer  = Adam

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=float(config['scheduler']['factor']), 
    patience=int(config['scheduler']['patience']), 
    threshold=1e-4, 
    threshold_mode='rel', 
    cooldown=0, 
    min_lr=float(config['scheduler']['min_lr']), 
    eps=1e-08)

best_val_loss = float('inf') #initialisation of val_loss = infinite at start

best_model_path = "Your_Model.pt"

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
    train_loss_eikonal = 0.0
    grad_norm_total = 0.0
    grad_norm_squared_total = 0.0
    num_points_total = 0

    num_batches = 0

    with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
        for batch in pbar:

          points = batch['points'].to(device)
          points_detached = points.clone().detach().requires_grad_(True)
          sdf = batch['sdf'].to(device)
          Cd = batch['Cd'].to(device)

          sdf_pred, cd_pred, z = model(points, sdf)

          loss_sdf = clamped_l1_loss(sdf_pred, sdf, delta=0.1) #loss L1 clamped between sdf_pred and sdf (GT)
          
          B, N, _ = points.shape
          idx = torch.randperm(N)[:N // 100] 
            
          eikonal_points = points_detached[:, idx, :].clone().detach().requires_grad_(True)
          sdf_pred_eikonal,_ = model.decoder(eikonal_points, z)  # z : latent vector

          # Eikonal Loss
          grad_outputs = torch.ones_like(sdf_pred_eikonal, device=device)
          gradients = torch.autograd.grad(
                outputs=sdf_pred_eikonal,
                inputs=eikonal_points,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
                )[0]
          grad_norm = torch.norm(gradients, dim=-1)
          loss_eikonal = ((grad_norm - 1.0) ** 2).mean()

          loss_cd = F.mse_loss(cd_pred.squeeze(-1), Cd) # MSE loss between Cd_pred and Cd (GT)
          loss = loss_sdf + lmbda * loss_cd + lambda_eikonal * loss_eikonal 

        # gradient descent

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          train_loss_sdf += loss_sdf.item()
          train_loss_cd  += loss_cd.item()
          train_loss_eikonal += loss_eikonal.item()
          grad_norm_total += grad_norm.sum().item()
          grad_norm_squared_total += (grad_norm ** 2).sum().item()
          num_points_total += grad_norm.numel()

          num_batches += 1

          pbar.set_postfix({
                "train_sdf": loss_sdf.item(),
                "train_Cd":  loss_cd.item(),
                'train_eikonal' : loss_eikonal.item() 
            })

    train_loss_sdf /= num_batches
    train_loss_cd  /= num_batches
    train_loss_eikonal /=num_batches
    
    train_total_loss = train_loss_sdf + lmbda * train_loss_cd + lambda_eikonal * train_loss_eikonal

    mean_grad_norm = grad_norm_total / num_points_total

    ## Validation ## (no backpropagation)

    model.eval()
    val_loss_sdf = 0.0
    val_loss_cd = 0.0
    val_loss_eikonal = 0.0
    num_batches_val = 0

    for batch in val_loader:
        points = batch['points'].to(device)
        sdf = batch['sdf'].to(device)
        Cd = batch['Cd'].to(device)
        with torch.no_grad():
        
            sdf_pred, cd_pred,_ = model(points, sdf) # model's predictions

            loss_sdf = clamped_l1_loss(sdf_pred, sdf, delta)
            loss_cd = F.mse_loss(cd_pred.squeeze(-1), Cd)

        z = model.encoder(points, sdf)

        B, N, _ = points.shape
        idx = torch.randperm(N)[:N // 100]

        eikonal_points = points[:, idx, :].clone().detach().requires_grad_(True)

        sdf_pred_eikonal, _ = model.decoder(eikonal_points, z)

        grad_outputs = torch.ones_like(sdf_pred_eikonal, device=device)

        gradients = torch.autograd.grad(
            outputs=sdf_pred_eikonal,
            inputs=eikonal_points,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]

        grad_norm = torch.norm(gradients, dim=-1)
        loss_eikonal = ((grad_norm - 1.0) ** 2).mean()  

        val_loss_sdf += loss_sdf.item()
        val_loss_cd  += loss_cd.item()
        val_loss_eikonal += loss_eikonal.item()
        num_batches_val += 1  

    val_loss_sdf /= num_batches_val
    val_loss_cd  /= num_batches_val
    val_loss_eikonal /= num_batches_val
    val_total_loss = val_loss_sdf + lmbda * val_loss_cd

    if val_total_loss < best_val_loss:
      best_val_loss = val_total_loss
      torch.save(model.state_dict(), best_model_path)
      print(f"New best model saved: val_total_loss = {val_total_loss:.4f}, val_loss_cd = {val_loss_cd}, val_loss_sdf = {val_loss_sdf}")

    val_total_plot = val_loss_sdf + val_loss_cd

    wandb.log({
    "epoch": epoch,
    "train_loss_sdf": train_loss_sdf,
    "train_loss_Cd": train_loss_cd,
    "train_loss_eikonal" : train_loss_eikonal,
    "train_total_loss": train_total_loss,
    "val_loss_sdf": val_loss_sdf,
    "val_loss_Cd": val_loss_cd,
    "val_loss_eikonal": val_loss_eikonal,
    "val_total_loss": val_total_loss,
    "val_total_plot": val_total_plot,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "grad_norm_mean": mean_grad_norm
    })

    scheduler.step(val_total_loss)
    for param_group in optimizer.param_groups:
      print(f"Current LR: {param_group['lr']}")


