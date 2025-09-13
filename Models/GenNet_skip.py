import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

### -- Modèle -- ###

## Encodeur ##

class SDFEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_dim)   # (x, y, z, sdf) --> hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
 
    def forward(self, points, sdf):
        
        #points: (B, N, 3)
        #sdf: (B, N)
        
        x = torch.cat([points, sdf.unsqueeze(-1)], dim=-1)  # (B, N, 4)

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        z = self.fc3(h)               # (B, N, latent_dim)
        z = z.mean(dim=1)             # Pooling → (B, latent_dim)
        return z #retourne un vecteur latent de dimension latent_dim
    
## Décodeur ##

class SDFDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + 3, hidden_dim) #première couche fully connected letent_dim + 3 --> hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # couche fully-connected hidden_dim --> hidden_dim
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim + latent_dim + 3, hidden_dim) # skip connection: on réinjecte le tatent + vecteur position
        self.fc5 = nn.Linear(hidden_dim, hidden_dim) # couche fully-connected hidden_dim --> hidden_dim
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)

        # Double tête
        self.head_sdf = nn.Linear(hidden_dim, 1) # tête de sortie SDF: hidden_dim --> 1
        self.head_cd = nn.Linear(latent_dim, 1) #tête de sortie Cd
        # Pas besoin des points pour Cd juste vecteur latent qui encode la géométrie + physique

        self.dropout_cd = nn.Dropout(dropout)

    def forward(self, x, z): #x = batch de points 3D, z: batch de vecteurs latents
        # x: (batch_dim, N, 3) — N points 3D
        # z: (batch_dim, latent_dim)

        B, N, _ = x.shape # B = batch_dim, N = N_points_échantillonnage

        z_expanded = z.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent_dim)
        input = torch.cat([x, z_expanded], dim=-1)     # (B, N, latent+3)

        h = F.relu(self.fc1(input))          
        h = F.relu(self.fc2(h))       
        h = F.relu(self.fc3(h)) 

        h = torch.cat([h, input], dim=-1) # concaténation de la sortie h avec le latent + le vecteur position

        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))                       

        sdf = self.head_sdf(h).squeeze(-1)             # (B, N) calcul de la sortie géométrique
        
        cd = self.head_cd(self.dropout_cd(z))
        # (B,) calcul de la sortie physique (coefficient de traînée)

        return sdf, cd # sdf est une valeur pour chaque point de chaque sample du batch
                       # cd est une valeur scalaire pour chaque sample du batch
    
## Auto-encodeur = Encodeur + Décodeur ##

class AutoencoderSDF(nn.Module):
    def __init__(self, latent_dim, hidden_dim, dropout):
        super().__init__()
        self.encoder = SDFEncoder(latent_dim, hidden_dim)
        self.decoder = SDFDecoder(latent_dim, hidden_dim, dropout)

    def forward(self, points, sdf):
        z = self.encoder(points, sdf)
        sdf_pred, cd_pred = self.decoder(points, z)
        return sdf_pred, cd_pred, z
    
    def predict_cd_only(self, z):
        return self.decoder.head_cd(self.decoder.dropout_cd(z))

