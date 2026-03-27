"""
Clustering Module for Security Events
Applies various clustering algorithms on learned embeddings
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings


@dataclass
class ClusteringConfig:
    """Configuration for clustering"""
    method: str = 'kmeans'  # kmeans, dbscan, hdbscan, gmm, hierarchical
    n_clusters: int = 10
    # DBSCAN/HDBSCAN params
    eps: float = 0.5
    min_samples: int = 5
    min_cluster_size: int = 10
    # GMM params
    covariance_type: str = 'full'
    # General
    random_state: int = 42


class SecurityEventClusterer:
    """
    Clustering engine for security event embeddings
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self.model = None
        self.labels_ = None
        self.n_clusters_ = None
        self.metrics_ = {}
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit clustering model and return cluster labels"""
        
        if self.config.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
                n_init=10
            )
            self.labels_ = self.model.fit_predict(embeddings)
        
        elif self.config.method == 'dbscan':
            self.model = DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples
            )
            self.labels_ = self.model.fit_predict(embeddings)
        
        elif self.config.method == 'hdbscan':
            self.model = HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples
            )
            self.labels_ = self.model.fit_predict(embeddings)
        
        elif self.config.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.config.n_clusters,
                covariance_type=self.config.covariance_type,
                random_state=self.config.random_state
            )
            self.labels_ = self.model.fit_predict(embeddings)
        
        elif self.config.method == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.config.n_clusters
            )
            self.labels_ = self.model.fit_predict(embeddings)
        
        else:
            raise ValueError(f"Unknown clustering method: {self.config.method}")
        
        # Count actual clusters (excluding noise label -1)
        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Calculate metrics
        self._calculate_metrics(embeddings)
        
        return self.labels_
    
    def _calculate_metrics(self, embeddings: np.ndarray):
        """Calculate clustering quality metrics"""
        # Need at least 2 clusters for metrics
        if self.n_clusters_ < 2:
            self.metrics_ = {
                'n_clusters': self.n_clusters_,
                'n_noise': int(np.sum(self.labels_ == -1)),
            }
            return
        
        # Filter out noise points for metrics calculation
        mask = self.labels_ != -1
        if mask.sum() < 2:
            return
        
        embeddings_filtered = embeddings[mask]
        labels_filtered = self.labels_[mask]
        
        # Check we have enough unique labels after filtering
        if len(set(labels_filtered)) < 2:
            self.metrics_ = {
                'n_clusters': self.n_clusters_,
                'n_noise': int(np.sum(self.labels_ == -1)),
            }
            return
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                silhouette = silhouette_score(embeddings_filtered, labels_filtered)
            except:
                silhouette = None
            
            try:
                calinski = calinski_harabasz_score(embeddings_filtered, labels_filtered)
            except:
                calinski = None
            
            try:
                davies = davies_bouldin_score(embeddings_filtered, labels_filtered)
            except:
                davies = None
        
        self.metrics_ = {
            'n_clusters': self.n_clusters_,
            'n_noise': int(np.sum(self.labels_ == -1)),
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
        }
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get size of each cluster"""
        if self.labels_ is None:
            return {}
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers if available"""
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        elif hasattr(self.model, 'means_'):
            return self.model.means_
        return None


def find_optimal_clusters(
    embeddings: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 20,
    method: str = 'silhouette'
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Find optimal number of clusters using various metrics
    
    Args:
        embeddings: Feature embeddings
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        method: Metric to optimize ('silhouette', 'calinski', 'elbow')
    
    Returns:
        Optimal number of clusters and scores for each k
    """
    results = []
    
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        result = {'k': k, 'inertia': kmeans.inertia_}
        
        try:
            result['silhouette'] = silhouette_score(embeddings, labels)
        except:
            result['silhouette'] = None
        
        try:
            result['calinski'] = calinski_harabasz_score(embeddings, labels)
        except:
            result['calinski'] = None
        
        results.append(result)
    
    # Find optimal k based on method
    if method == 'silhouette':
        valid_results = [r for r in results if r['silhouette'] is not None]
        if valid_results:
            optimal_k = max(valid_results, key=lambda x: x['silhouette'])['k']
        else:
            optimal_k = min_clusters
    elif method == 'calinski':
        valid_results = [r for r in results if r['calinski'] is not None]
        if valid_results:
            optimal_k = max(valid_results, key=lambda x: x['calinski'])['k']
        else:
            optimal_k = min_clusters
    else:  # elbow method
        # Find elbow using second derivative of inertia
        inertias = [r['inertia'] for r in results]
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            optimal_idx = np.argmax(diffs2) + 1
            optimal_k = results[optimal_idx]['k']
        else:
            optimal_k = min_clusters
    
    return optimal_k, results


class ClusterAnalyzer:
    """
    Analyze and interpret clusters of security events
    """
    
    def __init__(self, labels: np.ndarray):
        self.labels = labels
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    def analyze_cluster(
        self,
        cluster_id: int,
        df,  # pandas DataFrame
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze a specific cluster to extract security insights
        """
        mask = self.labels == cluster_id
        cluster_df = df[mask]
        
        analysis = {
            'cluster_id': cluster_id,
            'size': int(mask.sum()),
            'percentage': float(mask.sum() / len(self.labels) * 100),
        }
        
        # Analyze categorical features
        categorical_cols = ['subsystem', 'action', 'severity', 'protocol', 'user']
        for col in categorical_cols:
            if col in cluster_df.columns:
                value_counts = cluster_df[col].value_counts().head(top_n)
                analysis[f'top_{col}'] = value_counts.to_dict()
        
        # Analyze ports
        for port_col in ['dest_port', 'source_port']:
            if port_col in cluster_df.columns:
                port_counts = cluster_df[port_col].value_counts().head(top_n)
                analysis[f'top_{port_col}'] = port_counts.to_dict()
        
        # Analyze IPs
        for ip_col in ['source_ip', 'dest_ip']:
            if ip_col in cluster_df.columns:
                ip_counts = cluster_df[ip_col].value_counts().head(top_n)
                analysis[f'top_{ip_col}'] = ip_counts.to_dict()
        
        # Analyze temporal patterns
        if 'timestamp' in cluster_df.columns:
            timestamps = cluster_df['timestamp'].dropna()
            if len(timestamps) > 0:
                analysis['temporal'] = {
                    'first_event': str(timestamps.min()),
                    'last_event': str(timestamps.max()),
                    'hour_distribution': timestamps.dt.hour.value_counts().to_dict(),
                }
        
        # Extract common content patterns
        if 'content' in cluster_df.columns:
            contents = cluster_df['content'].dropna()
            if len(contents) > 0:
                # Get most common words
                all_words = ' '.join(contents.astype(str)).lower().split()
                word_counts = {}
                for word in all_words:
                    if len(word) > 3:  # Filter short words
                        word_counts[word] = word_counts.get(word, 0) + 1
                top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                analysis['top_content_words'] = dict(top_words)
        
        return analysis
    
    def get_cluster_summary(self, df) -> List[Dict[str, Any]]:
        """Get summary of all clusters"""
        summaries = []
        for cluster_id in sorted(set(self.labels)):
            if cluster_id == -1:
                continue  # Skip noise
            summary = self.analyze_cluster(cluster_id, df)
            summaries.append(summary)
        return summaries
    
    def identify_anomalous_clusters(
        self,
        df,
        size_threshold: float = 0.01
    ) -> List[int]:
        """
        Identify potentially anomalous clusters based on:
        - Small size (rare events)
        - High severity concentration
        - Unusual port/IP patterns
        """
        anomalous = []
        total_size = len(self.labels)
        
        for cluster_id in set(self.labels):
            if cluster_id == -1:
                continue
            
            mask = self.labels == cluster_id
            cluster_size = mask.sum()
            
            # Small clusters might be anomalies
            if cluster_size / total_size < size_threshold:
                anomalous.append(cluster_id)
                continue
            
            cluster_df = df[mask]
            
            # Check for high severity concentration
            if 'severity' in cluster_df.columns:
                severity_counts = cluster_df['severity'].value_counts(normalize=True)
                high_severity = ['high', 'critical', 'emergency', 'alert']
                high_ratio = sum(
                    severity_counts.get(s, 0) for s in high_severity
                )
                if high_ratio > 0.5:
                    anomalous.append(cluster_id)
                    continue
        
        return list(set(anomalous))
