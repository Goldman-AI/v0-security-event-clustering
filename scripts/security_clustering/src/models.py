"""
Deep Learning Models for Security Event Clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder model"""
    input_dim: int
    hidden_dims: List[int] = None
    latent_dim: int = 32
    dropout: float = 0.2
    use_batch_norm: bool = True
    activation: str = 'relu'  # relu, leaky_relu, elu, gelu
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class Encoder(nn.Module):
    """Encoder network for autoencoder"""
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.latent_layer = nn.Linear(in_dim, config.latent_dim)
    
    def _get_activation(self) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        }
        return activations.get(self.config.activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        z = self.latent_layer(x)
        return z


class Decoder(nn.Module):
    """Decoder network for autoencoder"""
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        layers = []
        hidden_dims_reversed = list(reversed(config.hidden_dims))
        in_dim = config.latent_dim
        
        for hidden_dim in hidden_dims_reversed:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, config.input_dim)
    
    def _get_activation(self) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        }
        return activations.get(self.config.activation, nn.ReLU())
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.layers(z)
        x_reconstructed = self.output_layer(x)
        return x_reconstructed


class SecurityEventAutoencoder(nn.Module):
    """
    Autoencoder for learning security event representations
    The latent space can be used for clustering
    """
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space"""
        return self.decoder(z)


class VariationalEncoder(nn.Module):
    """Variational encoder for VAE"""
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, config.latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class SecurityEventVAE(nn.Module):
    """
    Variational Autoencoder for security events
    Better for learning meaningful latent representations
    """
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = VariationalEncoder(config)
        self.decoder = Decoder(config)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (use mean)"""
        mu, _ = self.encoder(x)
        return mu
    
    def loss_function(
        self,
        x: torch.Tensor,
        x_reconstructed: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 0.001
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE loss function with reconstruction and KL divergence"""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


class DeepEmbeddingClustering(nn.Module):
    """
    Deep Embedding Clustering (DEC) model
    Combines autoencoder with clustering objective
    """
    
    def __init__(
        self,
        config: AutoencoderConfig,
        n_clusters: int = 10,
        alpha: float = 1.0
    ):
        super().__init__()
        self.config = config
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Autoencoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Cluster centers (initialized after pretraining)
        self.cluster_centers = nn.Parameter(
            torch.zeros(n_clusters, config.latent_dim),
            requires_grad=True
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        q = self.soft_assignment(z)
        return x_reconstructed, z, q
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def soft_assignment(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignment using Student's t-distribution
        q_ij = (1 + ||z_i - c_j||^2 / alpha)^(-(alpha+1)/2) / sum_j(...)
        """
        # Compute distances to cluster centers
        distances = torch.cdist(z, self.cluster_centers, p=2).pow(2)
        
        # Student's t-distribution kernel
        q = (1 + distances / self.alpha).pow(-(self.alpha + 1) / 2)
        
        # Normalize
        q = q / q.sum(dim=1, keepdim=True)
        
        return q
    
    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution P from soft assignments Q
        p_ij = q_ij^2 / f_j / sum_j(q_ij^2 / f_j)
        where f_j = sum_i(q_ij)
        """
        f = q.sum(dim=0)
        p = q.pow(2) / f
        p = p / p.sum(dim=1, keepdim=True)
        return p
    
    def clustering_loss(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """KL divergence between P and Q"""
        return F.kl_div(q.log(), p, reduction='batchmean')
    
    def initialize_centers(self, z: torch.Tensor, method: str = 'kmeans'):
        """Initialize cluster centers from encoded data"""
        from sklearn.cluster import KMeans
        
        z_numpy = z.detach().cpu().numpy()
        
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
            kmeans.fit(z_numpy)
            centers = torch.from_numpy(kmeans.cluster_centers_).float()
        else:
            # Random initialization
            indices = torch.randperm(z.size(0))[:self.n_clusters]
            centers = z[indices].detach()
        
        self.cluster_centers.data = centers.to(z.device)


class ContrastiveEncoder(nn.Module):
    """
    Contrastive learning encoder for security events
    Uses SimCLR-style contrastive learning
    """
    
    def __init__(self, config: AutoencoderConfig, projection_dim: int = 64):
        super().__init__()
        self.config = config
        
        # Base encoder
        self.encoder = Encoder(config)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        p = self.projection(z)
        return z, F.normalize(p, dim=1)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    @staticmethod
    def contrastive_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.5
    ) -> torch.Tensor:
        """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss"""
        batch_size = z1.size(0)
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / temperature
        
        # Create labels (positive pairs are (i, i+batch_size) and (i+batch_size, i))
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim.masked_fill_(mask, float('-inf'))
        
        # Cross entropy loss
        loss = F.cross_entropy(sim, labels)
        
        return loss
