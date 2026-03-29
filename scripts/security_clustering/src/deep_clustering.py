"""
Deep Learning Clustering Models for Security Events
Pure neural network-based clustering without traditional ML algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class DeepClusteringConfig:
    """Configuration for deep clustering models"""
    input_dim: int
    hidden_dims: List[int] = None
    latent_dim: int = 32
    n_clusters: int = 10
    dropout: float = 0.2
    use_batch_norm: bool = True
    alpha: float = 1.0  # Student's t-distribution degree of freedom
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


# =============================================================================
# 1. Deep Embedded Clustering (DEC)
# =============================================================================

class DECEncoder(nn.Module):
    """Encoder for DEC"""
    
    def __init__(self, config: DeepClusteringConfig):
        super().__init__()
        layers = []
        in_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DECDecoder(nn.Module):
    """Decoder for DEC"""
    
    def __init__(self, config: DeepClusteringConfig):
        super().__init__()
        layers = []
        hidden_dims_rev = list(reversed(config.hidden_dims))
        in_dim = config.latent_dim
        
        for hidden_dim in hidden_dims_rev:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class ClusteringLayer(nn.Module):
    """
    Clustering layer that computes soft cluster assignments
    using Student's t-distribution as kernel
    """
    
    def __init__(self, n_clusters: int, latent_dim: int, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Cluster centers as learnable parameters
        self.cluster_centers = nn.Parameter(
            torch.Tensor(n_clusters, latent_dim)
        )
        nn.init.xavier_uniform_(self.cluster_centers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignment Q using Student's t-distribution
        q_ij = (1 + ||z_i - μ_j||² / α)^(-(α+1)/2) / Σ_j'(...)
        """
        # z: (batch, latent_dim), centers: (n_clusters, latent_dim)
        # Compute squared distances
        q = 1.0 / (1.0 + (torch.cdist(z, self.cluster_centers) ** 2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q
    
    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary target distribution P
        p_ij = (q_ij² / f_j) / Σ_j'(q_ij'² / f_j')
        where f_j = Σ_i q_ij
        """
        weight = q ** 2 / q.sum(dim=0)
        return (weight.T / weight.sum(dim=1)).T


class DEC(nn.Module):
    """
    Deep Embedded Clustering (DEC)
    
    Reference: Xie et al., "Unsupervised Deep Embedding for Clustering Analysis" (ICML 2016)
    
    Training process:
    1. Pretrain autoencoder with reconstruction loss
    2. Initialize cluster centers with k-means on latent space
    3. Fine-tune by minimizing KL divergence between Q and auxiliary distribution P
    """
    
    def __init__(self, config: DeepClusteringConfig):
        super().__init__()
        self.config = config
        self.encoder = DECEncoder(config)
        self.decoder = DECDecoder(config)
        self.clustering = ClusteringLayer(
            config.n_clusters, config.latent_dim, config.alpha
        )
        self.pretrained = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (reconstructed, latent, soft_assignments)"""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        q = self.clustering(z)
        return x_recon, z, q
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """Get hard cluster assignments"""
        z = self.encoder(x)
        q = self.clustering(z)
        return q.argmax(dim=1)
    
    def initialize_centers(self, z: torch.Tensor):
        """Initialize cluster centers using k-means"""
        from sklearn.cluster import KMeans
        
        z_np = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.config.n_clusters, n_init=20, random_state=42)
        kmeans.fit(z_np)
        
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        self.clustering.cluster_centers.data = centers.to(z.device)
        print(f"Initialized {self.config.n_clusters} cluster centers")


# =============================================================================
# 2. Improved Deep Embedded Clustering (IDEC)
# =============================================================================

class IDEC(DEC):
    """
    Improved Deep Embedded Clustering (IDEC)
    
    Reference: Guo et al., "Improved Deep Embedded Clustering with Local Structure Preservation" (IJCAI 2017)
    
    Adds reconstruction loss to DEC to preserve local structure
    Loss = clustering_loss + γ * reconstruction_loss
    """
    
    def __init__(self, config: DeepClusteringConfig, gamma: float = 0.1):
        super().__init__(config)
        self.gamma = gamma  # Weight for reconstruction loss
    
    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        IDEC loss = KL(P||Q) + γ * MSE(x, x_recon)
        """
        # Clustering loss: KL divergence
        clustering_loss = F.kl_div(q.log(), p, reduction='batchmean')
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total loss
        total_loss = clustering_loss + self.gamma * recon_loss
        
        return total_loss, clustering_loss, recon_loss


# =============================================================================
# 3. Variational Deep Embedding (VaDE)
# =============================================================================

class VaDE(nn.Module):
    """
    Variational Deep Embedding for Clustering (VaDE)
    
    Reference: Jiang et al., "Variational Deep Embedding" (IJCAI 2017)
    
    Uses VAE with Gaussian Mixture Model prior in latent space
    Each cluster is modeled as a Gaussian component
    """
    
    def __init__(self, config: DeepClusteringConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        layers = []
        in_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.encoder_body = nn.Sequential(*layers)
        
        # Mean and log-variance for VAE
        self.fc_mu = nn.Linear(in_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, config.latent_dim)
        
        # Decoder
        dec_layers = []
        hidden_dims_rev = list(reversed(config.hidden_dims))
        in_dim = config.latent_dim
        for hidden_dim in hidden_dims_rev:
            dec_layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                dec_layers.append(nn.BatchNorm1d(hidden_dim))
            dec_layers.append(nn.ReLU())
            in_dim = hidden_dim
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)
        
        # GMM parameters (cluster-specific)
        # π_c: cluster prior probabilities
        self.pi = nn.Parameter(torch.ones(config.n_clusters) / config.n_clusters)
        # μ_c: cluster means in latent space
        self.mu_c = nn.Parameter(torch.randn(config.n_clusters, config.latent_dim) * 0.05)
        # log(σ²_c): cluster log-variances
        self.logvar_c = nn.Parameter(torch.zeros(config.n_clusters, config.latent_dim))
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar
    
    def get_gamma(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute posterior cluster assignment probabilities γ_c
        γ_c ∝ π_c * N(z | μ_c, σ²_c)
        """
        # z: (batch, latent_dim)
        # mu_c: (n_clusters, latent_dim)
        
        pi = F.softmax(self.pi, dim=0)  # Normalize cluster priors
        
        # Compute log N(z | μ_c, σ²_c) for each cluster
        # log N = -0.5 * (log(2π) + log(σ²) + (z-μ)²/σ²)
        var_c = torch.exp(self.logvar_c)  # (n_clusters, latent_dim)
        
        # Expand dimensions for broadcasting
        z_expanded = z.unsqueeze(1)  # (batch, 1, latent_dim)
        mu_c_expanded = self.mu_c.unsqueeze(0)  # (1, n_clusters, latent_dim)
        var_c_expanded = var_c.unsqueeze(0)  # (1, n_clusters, latent_dim)
        
        # Log probability
        log_p = -0.5 * (
            np.log(2 * np.pi) +
            self.logvar_c.unsqueeze(0) +
            (z_expanded - mu_c_expanded) ** 2 / var_c_expanded
        )
        log_p = log_p.sum(dim=2)  # Sum over latent dimensions: (batch, n_clusters)
        
        # Add log prior
        log_p = log_p + torch.log(pi + 1e-10)
        
        # Softmax to get posterior
        gamma = F.softmax(log_p, dim=1)
        
        return gamma
    
    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """Get hard cluster assignments"""
        mu, _ = self.encode(x)
        gamma = self.get_gamma(mu)
        return gamma.argmax(dim=1)
    
    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VaDE ELBO loss
        L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        where p(z) = Σ_c π_c * N(z | μ_c, σ²_c)
        """
        batch_size = x.size(0)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # Get cluster assignment probabilities
        gamma = self.get_gamma(z)  # (batch, n_clusters)
        
        pi = F.softmax(self.pi, dim=0)
        var_c = torch.exp(self.logvar_c)
        
        # KL divergence terms
        # E_q[log q(z|x)]
        log_qz = -0.5 * (
            self.config.latent_dim * np.log(2 * np.pi) +
            logvar.sum(dim=1) +
            ((z - mu) ** 2 / torch.exp(logvar)).sum(dim=1)
        )
        
        # E_q[log p(c)]
        log_pc = torch.log(pi + 1e-10)
        
        # E_q[log p(z|c)] - weighted by gamma
        z_expanded = z.unsqueeze(1)
        mu_c_expanded = self.mu_c.unsqueeze(0)
        var_c_expanded = var_c.unsqueeze(0)
        logvar_c_expanded = self.logvar_c.unsqueeze(0)
        
        log_pzc = -0.5 * (
            self.config.latent_dim * np.log(2 * np.pi) +
            logvar_c_expanded.sum(dim=2) +
            ((z_expanded - mu_c_expanded) ** 2 / var_c_expanded).sum(dim=2)
        )  # (batch, n_clusters)
        
        # E_q[log q(c|z)]
        log_qc = torch.log(gamma + 1e-10)
        
        # ELBO components
        # E_q(c|z)[log p(c) + log p(z|c) - log q(c|z)]
        kl_loss = (gamma * (log_qc - log_pc - log_pzc)).sum(dim=1).mean()
        kl_loss = kl_loss + log_qz.mean()
        
        total_loss = recon_loss + kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def initialize_gmm(self, z: torch.Tensor):
        """Initialize GMM parameters using sklearn GMM"""
        from sklearn.mixture import GaussianMixture
        
        z_np = z.detach().cpu().numpy()
        gmm = GaussianMixture(
            n_components=self.config.n_clusters,
            covariance_type='diag',
            n_init=5,
            random_state=42
        )
        gmm.fit(z_np)
        
        self.mu_c.data = torch.from_numpy(gmm.means_).float().to(z.device)
        self.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).float().to(z.device))
        self.pi.data = torch.log(torch.from_numpy(gmm.weights_).float().to(z.device))
        
        print(f"Initialized GMM with {self.config.n_clusters} components")


# =============================================================================
# 4. Deep Clustering Network (DCN)
# =============================================================================

class DCN(nn.Module):
    """
    Deep Clustering Network (DCN)
    
    Reference: Yang et al., "Towards K-means-friendly Spaces" (ICML 2017)
    
    Jointly learns representations and k-means clustering
    Loss = reconstruction_loss + λ * k-means_loss
    """
    
    def __init__(self, config: DeepClusteringConfig, lambda_kmeans: float = 1.0):
        super().__init__()
        self.config = config
        self.lambda_kmeans = lambda_kmeans
        
        # Encoder
        self.encoder = DECEncoder(config)
        
        # Decoder
        self.decoder = DECDecoder(config)
        
        # Cluster centers
        self.cluster_centers = nn.Parameter(
            torch.randn(config.n_clusters, config.latent_dim) * 0.05
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        
        # Compute distances to cluster centers
        distances = torch.cdist(z, self.cluster_centers)
        
        return x_recon, z, distances
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        distances = torch.cdist(z, self.cluster_centers)
        return distances.argmin(dim=1)
    
    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DCN loss = reconstruction_loss + λ * k-means_loss
        k-means_loss = Σ_i min_c ||z_i - μ_c||²
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # K-means loss: minimum distance to any cluster center
        kmeans_loss = distances.min(dim=1)[0].mean()
        
        total_loss = recon_loss + self.lambda_kmeans * kmeans_loss
        
        return total_loss, recon_loss, kmeans_loss
    
    def initialize_centers(self, z: torch.Tensor):
        """Initialize cluster centers using k-means"""
        from sklearn.cluster import KMeans
        
        z_np = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.config.n_clusters, n_init=20, random_state=42)
        kmeans.fit(z_np)
        
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        self.cluster_centers.data = centers.to(z.device)
        print(f"Initialized {self.config.n_clusters} cluster centers")


# =============================================================================
# 5. Self-Supervised Contrastive Clustering
# =============================================================================

class ContrastiveClustering(nn.Module):
    """
    Contrastive Clustering with Instance and Cluster-level Contrastive Learning
    
    Reference: Li et al., "Contrastive Clustering" (AAAI 2021)
    
    Uses contrastive learning at both instance and cluster levels
    """
    
    def __init__(self, config: DeepClusteringConfig, temperature: float = 0.5):
        super().__init__()
        self.config = config
        self.temperature = temperature
        
        # Encoder backbone
        self.encoder = DECEncoder(config)
        
        # Instance projection head
        self.instance_projector = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
        # Cluster projection head
        self.cluster_projector = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.n_clusters),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_instance = F.normalize(self.instance_projector(z), dim=1)
        c = self.cluster_projector(z)
        return z, z_instance, c
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        c = self.cluster_projector(z)
        return c.argmax(dim=1)
    
    def instance_contrastive_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """Instance-level contrastive loss (NT-Xent)"""
        batch_size = z_i.size(0)
        
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Mask for positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim.masked_fill_(mask, float('-inf'))
        
        # Labels: positive pairs are (i, i+batch_size)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)
        
        loss = F.cross_entropy(sim, labels)
        return loss
    
    def cluster_contrastive_loss(
        self,
        c_i: torch.Tensor,
        c_j: torch.Tensor
    ) -> torch.Tensor:
        """Cluster-level contrastive loss"""
        # Aggregate cluster assignments across batch
        c_i_agg = c_i.mean(dim=0)  # (n_clusters,)
        c_j_agg = c_j.mean(dim=0)
        
        # Normalize
        c_i_agg = F.normalize(c_i_agg.unsqueeze(0), dim=1).squeeze()
        c_j_agg = F.normalize(c_j_agg.unsqueeze(0), dim=1).squeeze()
        
        # Similarity
        sim = torch.mm(c_i.t(), c_j) / self.temperature  # (n_clusters, n_clusters)
        
        # Diagonal should be maximized (same cluster across augmentations)
        labels = torch.arange(self.config.n_clusters).to(c_i.device)
        
        loss_i = F.cross_entropy(sim, labels)
        loss_j = F.cross_entropy(sim.t(), labels)
        
        return (loss_i + loss_j) / 2


# =============================================================================
# 6. Deep Clustering Trainer
# =============================================================================

class DeepClusteringTrainer:
    """
    Unified trainer for all deep clustering models
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.lr = learning_rate
        self.wd = weight_decay
        self.history = {}
    
    def _create_loader(self, data: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(data),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def pretrain_autoencoder(
        self,
        data: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 256
    ):
        """Pretrain autoencoder component"""
        print("=" * 50)
        print("Phase 1: Pretraining Autoencoder")
        print("=" * 50)
        
        loader = self._create_loader(data, batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        
        self.history['pretrain_loss'] = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                
                if isinstance(self.model, VaDE):
                    x_recon, z, mu, logvar = self.model(x)
                    loss = F.mse_loss(x_recon, x)
                else:
                    x_recon, z, _ = self.model(x)
                    loss = F.mse_loss(x_recon, x)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(loader)
            self.history['pretrain_loss'].append(avg_loss)
            print(f"Pretrain Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        # Initialize cluster centers
        print("\nInitializing cluster centers...")
        self.model.eval()
        with torch.no_grad():
            all_z = []
            for batch in loader:
                x = batch[0].to(self.device)
                if isinstance(self.model, VaDE):
                    mu, _ = self.model.encode(x)
                    all_z.append(mu)
                else:
                    z = self.model.encode(x)
                    all_z.append(z)
            all_z = torch.cat(all_z, dim=0)
            
            if isinstance(self.model, VaDE):
                self.model.initialize_gmm(all_z)
            else:
                self.model.initialize_centers(all_z)
    
    def train_dec(
        self,
        data: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        update_interval: int = 140,
        tol: float = 1e-3
    ):
        """Train DEC/IDEC model"""
        print("=" * 50)
        print("Phase 2: Deep Clustering Training")
        print("=" * 50)
        
        loader = self._create_loader(data, batch_size, shuffle=False)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr * 0.1, weight_decay=self.wd
        )
        
        self.history['clustering_loss'] = []
        if isinstance(self.model, IDEC):
            self.history['recon_loss'] = []
        
        # Get initial assignments
        self.model.eval()
        with torch.no_grad():
            all_q = []
            for batch in loader:
                x = batch[0].to(self.device)
                _, _, q = self.model(x)
                all_q.append(q)
            all_q = torch.cat(all_q, dim=0)
            y_pred_last = all_q.argmax(dim=1).cpu().numpy()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_cluster_loss = 0
            total_recon_loss = 0
            
            # Compute target distribution P
            self.model.eval()
            with torch.no_grad():
                all_q = []
                for batch in loader:
                    x = batch[0].to(self.device)
                    _, _, q = self.model(x)
                    all_q.append(q)
                all_q = torch.cat(all_q, dim=0)
                p = ClusteringLayer.target_distribution(all_q)
            
            # Check convergence
            y_pred = all_q.argmax(dim=1).cpu().numpy()
            delta = np.sum(y_pred != y_pred_last) / len(y_pred)
            y_pred_last = y_pred
            
            if epoch > 0 and delta < tol:
                print(f"Converged at epoch {epoch+1} (delta={delta:.6f} < tol={tol})")
                break
            
            # Training step
            self.model.train()
            p_idx = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                x = batch[0].to(self.device)
                batch_size_actual = x.size(0)
                p_batch = p[p_idx:p_idx + batch_size_actual].to(self.device)
                p_idx += batch_size_actual
                
                optimizer.zero_grad()
                x_recon, z, q = self.model(x)
                
                if isinstance(self.model, IDEC):
                    loss, cluster_loss, recon_loss = self.model.loss_function(
                        x, x_recon, q, p_batch
                    )
                    total_recon_loss += recon_loss.item()
                else:
                    # Standard DEC: only clustering loss
                    loss = F.kl_div(q.log(), p_batch, reduction='batchmean')
                    cluster_loss = loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_cluster_loss += cluster_loss.item()
                pbar.set_postfix({'loss': loss.item(), 'delta': delta})
            
            avg_loss = total_loss / len(loader)
            self.history['clustering_loss'].append(total_cluster_loss / len(loader))
            
            if isinstance(self.model, IDEC):
                self.history['recon_loss'].append(total_recon_loss / len(loader))
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}, Delta = {delta:.4f}")
    
    def train_vade(
        self,
        data: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256
    ):
        """Train VaDE model"""
        print("=" * 50)
        print("Phase 2: VaDE Clustering Training")
        print("=" * 50)
        
        loader = self._create_loader(data, batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr * 0.1, weight_decay=self.wd
        )
        
        self.history['total_loss'] = []
        self.history['recon_loss'] = []
        self.history['kl_loss'] = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_recon = 0
            total_kl = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                
                x_recon, z, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = self.model.loss_function(
                    x, x_recon, z, mu, logvar
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            n = len(loader)
            self.history['total_loss'].append(total_loss / n)
            self.history['recon_loss'].append(total_recon / n)
            self.history['kl_loss'].append(total_kl / n)
            
            print(f"Epoch {epoch+1}: Loss = {total_loss/n:.6f}, "
                  f"Recon = {total_recon/n:.6f}, KL = {total_kl/n:.6f}")
    
    def train_dcn(
        self,
        data: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256
    ):
        """Train DCN model"""
        print("=" * 50)
        print("Phase 2: DCN Training")
        print("=" * 50)
        
        loader = self._create_loader(data, batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr * 0.1, weight_decay=self.wd
        )
        
        self.history['total_loss'] = []
        self.history['recon_loss'] = []
        self.history['kmeans_loss'] = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_recon = 0
            total_kmeans = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                
                x_recon, z, distances = self.model(x)
                loss, recon_loss, kmeans_loss = self.model.loss_function(
                    x, x_recon, z, distances
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kmeans += kmeans_loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            n = len(loader)
            self.history['total_loss'].append(total_loss / n)
            self.history['recon_loss'].append(total_recon / n)
            self.history['kmeans_loss'].append(total_kmeans / n)
            
            print(f"Epoch {epoch+1}: Loss = {total_loss/n:.6f}, "
                  f"Recon = {total_recon/n:.6f}, K-means = {total_kmeans/n:.6f}")
    
    def get_cluster_assignments(self, data: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
        """Get cluster assignments for data"""
        self.model.eval()
        loader = self._create_loader(data, batch_size, shuffle=False)
        
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                labels = self.model.get_cluster_assignments(x)
                all_labels.append(labels.cpu().numpy())
        
        return np.concatenate(all_labels)
    
    def get_embeddings(self, data: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
        """Get latent embeddings"""
        self.model.eval()
        loader = self._create_loader(data, batch_size, shuffle=False)
        
        all_z = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                if isinstance(self.model, VaDE):
                    z, _ = self.model.encode(x)
                else:
                    z = self.model.encode(x)
                all_z.append(z.cpu().numpy())
        
        return np.vstack(all_z)
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'config': self.model.config if hasattr(self.model, 'config') else None
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        print(f"Model loaded from {path}")
