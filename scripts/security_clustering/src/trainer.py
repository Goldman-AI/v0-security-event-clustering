"""
Training Module for Security Event Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import os

from .models import (
    SecurityEventAutoencoder,
    SecurityEventVAE,
    DeepEmbeddingClustering,
    AutoencoderConfig
)


@dataclass
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10  # Early stopping patience
    min_delta: float = 1e-4  # Minimum improvement for early stopping
    checkpoint_dir: str = 'checkpoints'
    device: str = 'auto'  # auto, cpu, cuda, mps
    
    # VAE specific
    kl_weight: float = 0.001
    kl_annealing: bool = True
    kl_annealing_epochs: int = 20
    
    # DEC specific
    dec_update_interval: int = 140  # Update target distribution every N batches
    dec_clustering_weight: float = 0.1


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class SecurityEventTrainer:
    """
    Trainer for security event deep learning models
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None
    ):
        self.config = config or TrainingConfig()
        self.model = model
        
        # Set device
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.config.device)
        
        self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': []}
    
    def _create_optimizer(self):
        """Create optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def _create_dataloader(
        self,
        data: torch.Tensor,
        shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader from tensor"""
        dataset = TensorDataset(data)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def train_autoencoder(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """Train standard autoencoder"""
        self._create_optimizer()
        train_loader = self._create_dataloader(train_data)
        val_loader = self._create_dataloader(val_data, shuffle=False) if val_data is not None else None
        
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        
        criterion = nn.MSELoss()
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
            for batch in pbar:
                x = batch[0].to(self.device)
                
                self.optimizer.zero_grad()
                x_reconstructed, _ = self.model(x)
                loss = criterion(x_reconstructed, x)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader:
                val_loss = self._validate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
                
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}')
                
                if early_stopping(val_loss):
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}')
            
            if callback:
                callback(epoch, self.history)
        
        return self.history
    
    def train_vae(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """Train Variational Autoencoder"""
        if not isinstance(self.model, SecurityEventVAE):
            raise ValueError("Model must be SecurityEventVAE for VAE training")
        
        self._create_optimizer()
        train_loader = self._create_dataloader(train_data)
        val_loader = self._create_dataloader(val_data, shuffle=False) if val_data is not None else None
        
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        
        self.history['recon_loss'] = []
        self.history['kl_loss'] = []
        
        for epoch in range(self.config.epochs):
            # KL annealing
            if self.config.kl_annealing:
                kl_weight = min(
                    self.config.kl_weight,
                    self.config.kl_weight * (epoch + 1) / self.config.kl_annealing_epochs
                )
            else:
                kl_weight = self.config.kl_weight
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_recon = 0.0
            train_kl = 0.0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
            for batch in pbar:
                x = batch[0].to(self.device)
                
                self.optimizer.zero_grad()
                x_reconstructed, z, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = self.model.loss_function(
                    x, x_reconstructed, mu, logvar, kl_weight
                )
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'recon': recon_loss.item(),
                    'kl': kl_loss.item()
                })
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            self.history['recon_loss'].append(train_recon / len(train_loader))
            self.history['kl_loss'].append(train_kl / len(train_loader))
            
            # Validation
            if val_loader:
                val_loss = self._validate_vae(val_loader, kl_weight)
                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
                
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}, KL Weight = {kl_weight:.6f}')
                
                if early_stopping(val_loss):
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}')
            
            if callback:
                callback(epoch, self.history)
        
        return self.history
    
    def train_dec(
        self,
        train_data: torch.Tensor,
        pretrain_epochs: int = 50,
        callback: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """Train Deep Embedding Clustering model"""
        if not isinstance(self.model, DeepEmbeddingClustering):
            raise ValueError("Model must be DeepEmbeddingClustering for DEC training")
        
        # Phase 1: Pretrain autoencoder
        print("Phase 1: Pretraining autoencoder...")
        self._create_optimizer()
        train_loader = self._create_dataloader(train_data)
        
        criterion = nn.MSELoss()
        
        for epoch in range(pretrain_epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f'Pretrain {epoch+1}/{pretrain_epochs}'):
                x = batch[0].to(self.device)
                
                self.optimizer.zero_grad()
                z = self.model.encoder(x)
                x_reconstructed = self.model.decoder(z)
                loss = criterion(x_reconstructed, x)
                loss.backward()
                
                self.optimizer.step()
                train_loss += loss.item()
            
            print(f'Pretrain Epoch {epoch+1}: Loss = {train_loss/len(train_loader):.6f}')
        
        # Initialize cluster centers
        print("Initializing cluster centers...")
        self.model.eval()
        with torch.no_grad():
            all_z = []
            for batch in train_loader:
                x = batch[0].to(self.device)
                z = self.model.encoder(x)
                all_z.append(z)
            all_z = torch.cat(all_z, dim=0)
            self.model.initialize_centers(all_z)
        
        # Phase 2: Fine-tune with clustering objective
        print("Phase 2: Fine-tuning with clustering objective...")
        self._create_optimizer()  # Reset optimizer
        
        self.history['clustering_loss'] = []
        self.history['recon_loss'] = []
        
        batch_count = 0
        target_p = None
        
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_recon_loss = 0.0
            epoch_cluster_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}'):
                x = batch[0].to(self.device)
                
                # Update target distribution periodically
                if batch_count % self.config.dec_update_interval == 0:
                    self.model.eval()
                    with torch.no_grad():
                        _, _, q = self.model(x)
                        target_p = self.model.target_distribution(q)
                    self.model.train()
                
                self.optimizer.zero_grad()
                
                x_reconstructed, z, q = self.model(x)
                
                # Reconstruction loss
                recon_loss = criterion(x_reconstructed, x)
                
                # Clustering loss
                if target_p is not None:
                    p = self.model.target_distribution(q)
                    cluster_loss = self.model.clustering_loss(q, p)
                else:
                    cluster_loss = torch.tensor(0.0).to(self.device)
                
                # Combined loss
                loss = recon_loss + self.config.dec_clustering_weight * cluster_loss
                loss.backward()
                
                self.optimizer.step()
                
                epoch_recon_loss += recon_loss.item()
                epoch_cluster_loss += cluster_loss.item()
                batch_count += 1
            
            avg_recon = epoch_recon_loss / len(train_loader)
            avg_cluster = epoch_cluster_loss / len(train_loader)
            
            self.history['train_loss'].append(avg_recon + avg_cluster)
            self.history['recon_loss'].append(avg_recon)
            self.history['clustering_loss'].append(avg_cluster)
            
            print(f'Epoch {epoch+1}: Recon Loss = {avg_recon:.6f}, Cluster Loss = {avg_cluster:.6f}')
            
            if callback:
                callback(epoch, self.history)
        
        return self.history
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate autoencoder"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                x_reconstructed, _ = self.model(x)
                loss = criterion(x_reconstructed, x)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def _validate_vae(self, val_loader: DataLoader, kl_weight: float) -> float:
        """Validate VAE"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                x_reconstructed, z, mu, logvar = self.model(x)
                loss, _, _ = self.model.loss_function(x, x_reconstructed, mu, logvar, kl_weight)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def get_embeddings(self, data: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
        """Extract embeddings from trained model"""
        self.model.eval()
        embeddings = []
        
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                z = self.model.encode(x)
                embeddings.append(z.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history,
            'config': self.config,
        }, path)
        print(f'Model saved to {path}')
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        if checkpoint.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model loaded from {path}')
