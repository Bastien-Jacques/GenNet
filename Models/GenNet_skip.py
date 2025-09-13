import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

### -- Model -- ###

## Encoder ##

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
        return z (,latent_dim)
    
## Decoder ##

class SDFDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + 3, hidden_dim) #first fully connected letent_dim + 3 --> hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # fully-connected hidden_dim --> hidden_dim
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim + latent_dim + 3, hidden_dim) # skip connection
        self.fc5 = nn.Linear(hidden_dim, hidden_dim) # fully-connected hidden_dim --> hidden_dim
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)

        # Double head
        self.head_sdf = nn.Linear(hidden_dim, 1) # SDF head: hidden_dim --> 1
        self.head_cd = nn.Linear(latent_dim, 1) #Cd head: latent_dim --> 1
        self.dropout_cd = nn.Dropout(dropout)

    def forward(self, x, z): #x = batch de points 3D, z: batch de vecteurs latents
        # x: (batch_dim, N, 3) — N points 3D
        # z: (batch_dim, latent_dim)

        B, N, _ = x.shape # B = batch_dim, N = amount of points of SDF

        z_expanded = z.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent_dim)
        input = torch.cat([x, z_expanded], dim=-1)     # (B, N, latent+3)

        h = F.relu(self.fc1(input))          
        h = F.relu(self.fc2(h))       
        h = F.relu(self.fc3(h)) 

        h = torch.cat([h, input], dim=-1) # concatenation between h and latent + position vector

        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))                       

        sdf = self.head_sdf(h).squeeze(-1)             # (B, N)
        cd = self.head_cd(self.dropout_cd(z))
        # (B,)

        return sdf, cd 
    
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

