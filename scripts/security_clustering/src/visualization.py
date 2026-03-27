"""
Visualization Module for Security Event Clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
from sklearn.manifold import TSNE
import warnings


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = 'umap',
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce high-dimensional embeddings for visualization
    
    Args:
        embeddings: High-dimensional embeddings
        method: Reduction method ('umap', 'tsne')
        n_components: Target dimensions
        random_state: Random seed
    
    Returns:
        Reduced embeddings
    """
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean'
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            method = 'tsne'
    
    if method == 'tsne':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=min(30, len(embeddings) - 1),
                n_iter=1000
            )
            return reducer.fit_transform(embeddings)
    
    raise ValueError(f"Unknown method: {method}")


def plot_clusters(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Security Event Clusters",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show_legend: bool = True,
    alpha: float = 0.6,
    point_size: int = 20
) -> plt.Figure:
    """
    Plot clustered embeddings in 2D
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    
    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        
        if label == -1:
            # Noise points
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c='gray',
                s=point_size // 2,
                alpha=alpha / 2,
                label='Noise'
            )
        else:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx % len(colors)]],
                s=point_size,
                alpha=alpha,
                label=f'Cluster {label}'
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    
    if show_legend and n_clusters <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_cluster_distribution(
    labels: np.ndarray,
    title: str = "Cluster Size Distribution",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of cluster sizes
    """
    # Count cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    
    # Separate noise if present
    noise_mask = unique == -1
    if noise_mask.any():
        noise_count = counts[noise_mask][0]
        unique = unique[~noise_mask]
        counts = counts[~noise_mask]
    else:
        noise_count = 0
    
    # Sort by size
    sort_idx = np.argsort(counts)[::-1]
    unique = unique[sort_idx]
    counts = counts[sort_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(counts)))
    axes[0].bar(range(len(counts)), counts, color=colors)
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('Number of Events', fontsize=12)
    axes[0].set_title('Cluster Sizes', fontsize=14)
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels([f'C{c}' for c in unique], rotation=45)
    
    # Pie chart for top clusters
    top_n = min(10, len(counts))
    top_counts = counts[:top_n]
    top_labels = [f'Cluster {c}' for c in unique[:top_n]]
    
    if len(counts) > top_n:
        other_count = counts[top_n:].sum()
        top_counts = np.append(top_counts, other_count)
        top_labels.append('Other')
    
    if noise_count > 0:
        top_counts = np.append(top_counts, noise_count)
        top_labels.append('Noise')
    
    axes[1].pie(top_counts, labels=top_labels, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Event Distribution', fontsize=14)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training loss history
    """
    n_plots = len(history)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0], figsize[1]))
    
    if n_plots == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values, label=name, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(name.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_cluster_analysis(
    cluster_summaries: List[Dict[str, Any]],
    feature: str = 'subsystem',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot analysis of a specific feature across clusters
    """
    n_clusters = len(cluster_summaries)
    
    if n_clusters == 0:
        print("No clusters to analyze")
        return None
    
    # Collect feature data
    feature_key = f'top_{feature}'
    all_values = set()
    cluster_data = []
    
    for summary in cluster_summaries:
        feature_dict = summary.get(feature_key, {})
        all_values.update(feature_dict.keys())
        cluster_data.append(feature_dict)
    
    if not all_values:
        print(f"No data for feature: {feature}")
        return None
    
    # Create matrix
    all_values = sorted(all_values)[:15]  # Limit to top 15 values
    matrix = np.zeros((n_clusters, len(all_values)))
    
    for i, data in enumerate(cluster_data):
        for j, val in enumerate(all_values):
            matrix[i, j] = data.get(val, 0)
    
    # Normalize by row (cluster)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        matrix_norm,
        xticklabels=all_values,
        yticklabels=[f"Cluster {s['cluster_id']}" for s in cluster_summaries],
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        ax=ax
    )
    
    ax.set_title(title or f'{feature.title()} Distribution Across Clusters', fontsize=14, fontweight='bold')
    ax.set_xlabel(feature.title(), fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_temporal_patterns(
    cluster_summaries: List[Dict[str, Any]],
    title: str = "Temporal Patterns by Cluster",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot temporal patterns (hour of day) across clusters
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    hours = list(range(24))
    
    for summary in cluster_summaries:
        temporal = summary.get('temporal', {})
        hour_dist = temporal.get('hour_distribution', {})
        
        if hour_dist:
            counts = [hour_dist.get(h, 0) for h in hours]
            total = sum(counts)
            if total > 0:
                counts = [c / total for c in counts]
                ax.plot(hours, counts, marker='o', label=f"Cluster {summary['cluster_id']}", alpha=0.7)
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Proportion of Events', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_elbow_curve(
    results: List[Dict[str, Any]],
    title: str = "Optimal Cluster Selection",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot elbow curve and silhouette scores for cluster selection
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    ks = [r['k'] for r in results]
    
    # Inertia (elbow)
    inertias = [r['inertia'] for r in results]
    axes[0].plot(ks, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette
    silhouettes = [r.get('silhouette') for r in results]
    valid_sil = [(k, s) for k, s in zip(ks, silhouettes) if s is not None]
    if valid_sil:
        axes[1].plot([k for k, _ in valid_sil], [s for _, s in valid_sil], 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score', fontsize=12)
        axes[1].grid(True, alpha=0.3)
    
    # Calinski-Harabasz
    calinskis = [r.get('calinski') for r in results]
    valid_cal = [(k, c) for k, c in zip(ks, calinskis) if c is not None]
    if valid_cal:
        axes[2].plot([k for k, _ in valid_cal], [c for _, c in valid_cal], 'ro-', linewidth=2, markersize=8)
        axes[2].set_xlabel('Number of Clusters', fontsize=12)
        axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=12)
        axes[2].set_title('Calinski-Harabasz Score', fontsize=12)
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def create_cluster_report(
    cluster_summaries: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: str = "cluster_report.txt"
):
    """
    Create a text report of clustering results
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SECURITY EVENT CLUSTERING REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Overall metrics
    lines.append("CLUSTERING METRICS")
    lines.append("-" * 40)
    for key, value in metrics.items():
        if value is not None:
            lines.append(f"  {key}: {value}")
    lines.append("")
    
    # Cluster details
    lines.append("CLUSTER DETAILS")
    lines.append("-" * 40)
    
    for summary in sorted(cluster_summaries, key=lambda x: x['size'], reverse=True):
        lines.append(f"\nCluster {summary['cluster_id']}:")
        lines.append(f"  Size: {summary['size']} events ({summary['percentage']:.1f}%)")
        
        # Top subsystems
        if 'top_subsystem' in summary:
            top_sub = list(summary['top_subsystem'].items())[:3]
            lines.append(f"  Top Subsystems: {', '.join([f'{k}({v})' for k, v in top_sub])}")
        
        # Top severities
        if 'top_severity' in summary:
            top_sev = list(summary['top_severity'].items())[:3]
            lines.append(f"  Top Severities: {', '.join([f'{k}({v})' for k, v in top_sev])}")
        
        # Top ports
        if 'top_dest_port' in summary:
            top_ports = list(summary['top_dest_port'].items())[:5]
            lines.append(f"  Top Dest Ports: {', '.join([f'{int(k)}({v})' for k, v in top_ports])}")
        
        # Top content words
        if 'top_content_words' in summary:
            top_words = list(summary['top_content_words'].items())[:5]
            lines.append(f"  Key Terms: {', '.join([w for w, _ in top_words])}")
    
    lines.append("")
    lines.append("=" * 60)
    
    report = '\n'.join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")
    return report
