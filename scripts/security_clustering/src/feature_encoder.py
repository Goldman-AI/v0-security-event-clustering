"""
Feature Encoder Module
Handles encoding of security event features for deep learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import pickle
from dataclasses import dataclass
import torch


@dataclass
class EncoderConfig:
    """Configuration for feature encoding"""
    max_vocab_size: int = 10000  # Maximum vocabulary size for categorical features
    min_freq: int = 2  # Minimum frequency for vocabulary inclusion
    max_content_tokens: int = 50  # Maximum tokens to consider in content field
    embedding_dim: int = 32  # Embedding dimension for categorical features
    unknown_token: str = '<UNK>'
    padding_token: str = '<PAD>'


class FeatureEncoder:
    """Encodes security event features into numerical representations"""
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        
        # Vocabularies for categorical features
        self.vocabularies: Dict[str, Dict[str, int]] = {}
        self.reverse_vocabularies: Dict[str, Dict[int, str]] = {}
        
        # Statistics for numerical features
        self.numerical_stats: Dict[str, Dict[str, float]] = {}
        
        # Content tokenizer vocabulary
        self.content_vocab: Dict[str, int] = {}
        
        # Feature dimensions
        self.feature_dims: Dict[str, int] = {}
        
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEncoder':
        """Fit the encoder on training data"""
        
        # Define feature types
        categorical_features = ['subsystem', 'user', 'action', 'severity', 'protocol']
        numerical_features = ['source_port', 'dest_port']
        ip_features = ['source_ip', 'dest_ip']
        text_features = ['content']
        
        # Fit categorical features
        for feature in categorical_features:
            if feature in df.columns:
                self._fit_categorical(df[feature].fillna(''), feature)
        
        # Fit numerical features
        for feature in numerical_features:
            if feature in df.columns:
                self._fit_numerical(df[feature].fillna(0), feature)
        
        # Fit IP features
        for feature in ip_features:
            if feature in df.columns:
                self._fit_ip(df[feature].fillna(''), feature)
        
        # Fit text features
        for feature in text_features:
            if feature in df.columns:
                self._fit_text(df[feature].fillna(''), feature)
        
        # Calculate total feature dimension
        self._calculate_feature_dims()
        
        self._is_fitted = True
        return self
    
    def _fit_categorical(self, series: pd.Series, name: str):
        """Fit vocabulary for categorical feature"""
        counter = Counter(series.astype(str))
        
        # Filter by minimum frequency and max vocab size
        vocab_items = [
            (item, count) for item, count in counter.most_common()
            if count >= self.config.min_freq
        ][:self.config.max_vocab_size - 2]  # Reserve space for special tokens
        
        # Build vocabulary
        vocab = {
            self.config.padding_token: 0,
            self.config.unknown_token: 1,
        }
        for item, _ in vocab_items:
            vocab[item] = len(vocab)
        
        self.vocabularies[name] = vocab
        self.reverse_vocabularies[name] = {v: k for k, v in vocab.items()}
    
    def _fit_numerical(self, series: pd.Series, name: str):
        """Fit statistics for numerical feature"""
        values = pd.to_numeric(series, errors='coerce').fillna(0)
        self.numerical_stats[name] = {
            'mean': float(values.mean()),
            'std': float(values.std()) + 1e-8,  # Avoid division by zero
            'min': float(values.min()),
            'max': float(values.max()),
        }
    
    def _fit_ip(self, series: pd.Series, name: str):
        """Fit IP address encoding (just mark as fitted)"""
        # IP addresses are encoded as 4 normalized octets
        self.feature_dims[name] = 4
    
    def _fit_text(self, series: pd.Series, name: str):
        """Fit vocabulary for text content"""
        all_tokens = []
        for text in series.astype(str):
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
        
        counter = Counter(all_tokens)
        vocab_items = [
            (item, count) for item, count in counter.most_common()
            if count >= self.config.min_freq
        ][:self.config.max_vocab_size - 2]
        
        vocab = {
            self.config.padding_token: 0,
            self.config.unknown_token: 1,
        }
        for item, _ in vocab_items:
            vocab[item] = len(vocab)
        
        self.content_vocab = vocab
        self.feature_dims[f'{name}_vocab_size'] = len(vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for security event content"""
        # Convert to lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens[:self.config.max_content_tokens]
    
    def _calculate_feature_dims(self):
        """Calculate dimensions for each feature type"""
        self.feature_dims['categorical_total'] = sum(
            len(v) for v in self.vocabularies.values()
        )
        self.feature_dims['numerical_total'] = len(self.numerical_stats)
        self.feature_dims['ip_total'] = 8  # 4 octets for src + 4 for dst
        self.feature_dims['temporal_total'] = 4  # hour, day_of_week, day_of_month, month
    
    def get_total_dim(self) -> int:
        """Get total dimension of encoded features"""
        # Categorical (one-hot for each)
        cat_dim = sum(len(v) for v in self.vocabularies.values())
        # Numerical (normalized values)
        num_dim = len(self.numerical_stats)
        # IP (4 octets each, normalized)
        ip_dim = 8
        # Temporal
        temporal_dim = 4
        # Content (bag of words)
        content_dim = len(self.content_vocab) if self.content_vocab else 0
        
        return cat_dim + num_dim + ip_dim + temporal_dim + content_dim
    
    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Transform DataFrame to tensor"""
        if not self._is_fitted:
            raise RuntimeError("Encoder must be fitted before transform")
        
        n_samples = len(df)
        
        # Collect all feature vectors
        features = []
        
        # Encode categorical features (one-hot)
        for name, vocab in self.vocabularies.items():
            if name in df.columns:
                encoded = self._encode_categorical(df[name].fillna(''), vocab)
                features.append(encoded)
        
        # Encode numerical features
        for name, stats in self.numerical_stats.items():
            if name in df.columns:
                encoded = self._encode_numerical(df[name].fillna(0), stats)
                features.append(encoded.reshape(-1, 1))
        
        # Encode IP features
        for ip_col in ['source_ip', 'dest_ip']:
            if ip_col in df.columns:
                encoded = self._encode_ip(df[ip_col].fillna(''))
                features.append(encoded)
        
        # Encode temporal features
        if 'timestamp' in df.columns:
            encoded = self._encode_temporal(df['timestamp'])
            features.append(encoded)
        
        # Encode content (bag of words)
        if 'content' in df.columns and self.content_vocab:
            encoded = self._encode_content(df['content'].fillna(''))
            features.append(encoded)
        
        # Concatenate all features
        if features:
            feature_matrix = np.hstack(features)
        else:
            feature_matrix = np.zeros((n_samples, 1))
        
        return torch.FloatTensor(feature_matrix)
    
    def _encode_categorical(self, series: pd.Series, vocab: Dict[str, int]) -> np.ndarray:
        """One-hot encode categorical feature"""
        n_samples = len(series)
        vocab_size = len(vocab)
        encoded = np.zeros((n_samples, vocab_size), dtype=np.float32)
        
        for i, value in enumerate(series.astype(str)):
            idx = vocab.get(value, vocab.get(self.config.unknown_token, 1))
            encoded[i, idx] = 1.0
        
        return encoded
    
    def _encode_numerical(self, series: pd.Series, stats: Dict[str, float]) -> np.ndarray:
        """Normalize numerical feature"""
        values = pd.to_numeric(series, errors='coerce').fillna(0).values
        normalized = (values - stats['mean']) / stats['std']
        return normalized.astype(np.float32)
    
    def _encode_ip(self, series: pd.Series) -> np.ndarray:
        """Encode IP addresses as normalized octets"""
        n_samples = len(series)
        encoded = np.zeros((n_samples, 4), dtype=np.float32)
        
        for i, ip in enumerate(series.astype(str)):
            if ip and ip != 'nan':
                try:
                    octets = ip.split('.')
                    for j, octet in enumerate(octets[:4]):
                        encoded[i, j] = int(octet) / 255.0
                except (ValueError, IndexError):
                    pass
        
        return encoded
    
    def _encode_temporal(self, series: pd.Series) -> np.ndarray:
        """Encode temporal features"""
        n_samples = len(series)
        encoded = np.zeros((n_samples, 4), dtype=np.float32)
        
        for i, ts in enumerate(series):
            if pd.notna(ts):
                try:
                    if isinstance(ts, str):
                        ts = pd.to_datetime(ts)
                    encoded[i, 0] = ts.hour / 24.0  # Hour normalized
                    encoded[i, 1] = ts.dayofweek / 7.0  # Day of week
                    encoded[i, 2] = ts.day / 31.0  # Day of month
                    encoded[i, 3] = ts.month / 12.0  # Month
                except:
                    pass
        
        return encoded
    
    def _encode_content(self, series: pd.Series) -> np.ndarray:
        """Bag of words encoding for content"""
        n_samples = len(series)
        vocab_size = len(self.content_vocab)
        encoded = np.zeros((n_samples, vocab_size), dtype=np.float32)
        
        for i, text in enumerate(series.astype(str)):
            tokens = self._tokenize(text)
            for token in tokens:
                idx = self.content_vocab.get(token, self.content_vocab.get(self.config.unknown_token, 1))
                encoded[i, idx] += 1.0
        
        # Normalize by document length
        row_sums = encoded.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        encoded = encoded / row_sums
        
        return encoded
    
    def save(self, path: str):
        """Save encoder state"""
        state = {
            'config': self.config,
            'vocabularies': self.vocabularies,
            'reverse_vocabularies': self.reverse_vocabularies,
            'numerical_stats': self.numerical_stats,
            'content_vocab': self.content_vocab,
            'feature_dims': self.feature_dims,
            '_is_fitted': self._is_fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEncoder':
        """Load encoder from file"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        encoder = cls(state['config'])
        encoder.vocabularies = state['vocabularies']
        encoder.reverse_vocabularies = state['reverse_vocabularies']
        encoder.numerical_stats = state['numerical_stats']
        encoder.content_vocab = state['content_vocab']
        encoder.feature_dims = state['feature_dims']
        encoder._is_fitted = state['_is_fitted']
        
        return encoder
