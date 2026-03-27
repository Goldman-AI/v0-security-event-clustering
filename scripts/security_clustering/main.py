"""
Security Event Clustering - Main Script
Deep learning based clustering of security events using PyTorch
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser import SecurityEventParser
from src.feature_encoder import FeatureEncoder, EncoderConfig
from src.models import (
    SecurityEventAutoencoder,
    SecurityEventVAE,
    DeepEmbeddingClustering,
    AutoencoderConfig
)
from src.trainer import SecurityEventTrainer, TrainingConfig
from src.clustering import (
    SecurityEventClusterer,
    ClusteringConfig,
    ClusterAnalyzer,
    find_optimal_clusters
)
from src.visualization import (
    reduce_dimensions,
    plot_clusters,
    plot_cluster_distribution,
    plot_training_history,
    plot_cluster_analysis,
    plot_elbow_curve,
    create_cluster_report
)
from src.data_generator import SecurityEventGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Security Event Clustering using Deep Learning'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file containing security events (one per line)'
    )
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate synthetic data for demonstration'
    )
    parser.add_argument(
        '--n-events',
        type=int,
        default=5000,
        help='Number of events to generate (if using --generate-data)'
    )
    parser.add_argument(
        '--model-type',
        choices=['autoencoder', 'vae', 'dec'],
        default='vae',
        help='Type of model to use'
    )
    parser.add_argument(
        '--clustering',
        choices=['kmeans', 'dbscan', 'hdbscan', 'gmm'],
        default='kmeans',
        help='Clustering algorithm'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=0,
        help='Number of clusters (0 = auto-detect)'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=32,
        help='Latent space dimension'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for training'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device for training'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SECURITY EVENT CLUSTERING")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Load or Generate Data
    # =========================================================================
    print("\n[Step 1] Loading data...")
    
    parser = SecurityEventParser()
    
    if args.generate_data or not args.input:
        print(f"Generating {args.n_events} synthetic security events...")
        generator = SecurityEventGenerator(seed=42)
        events_raw = generator.generate_dataset(n_events=args.n_events)
        
        # Save generated data
        data_path = os.path.join(args.output_dir, 'generated_events.log')
        generator.save_dataset(events_raw, data_path)
    else:
        print(f"Loading events from {args.input}...")
        with open(args.input, 'r') as f:
            events_raw = [line.strip() for line in f if line.strip()]
    
    # Parse events
    print(f"Parsing {len(events_raw)} events...")
    events = parser.parse_lines(events_raw)
    df = parser.events_to_dataframe(events)
    
    print(f"\nDataset Summary:")
    print(f"  Total events: {len(df)}")
    print(f"  Subsystems: {df['subsystem'].nunique()}")
    print(f"  Unique source IPs: {df['source_ip'].nunique()}")
    print(f"  Unique dest IPs: {df['dest_ip'].nunique()}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # =========================================================================
    # Step 2: Feature Encoding
    # =========================================================================
    print("\n[Step 2] Encoding features...")
    
    encoder_config = EncoderConfig(
        max_vocab_size=5000,
        min_freq=2,
        max_content_tokens=50
    )
    encoder = FeatureEncoder(encoder_config)
    encoder.fit(df)
    
    # Transform to tensor
    X = encoder.transform(df)
    input_dim = X.shape[1]
    
    print(f"  Feature dimension: {input_dim}")
    print(f"  Tensor shape: {X.shape}")
    
    # Save encoder
    encoder_path = os.path.join(args.output_dir, 'encoder.pkl')
    encoder.save(encoder_path)
    print(f"  Encoder saved to {encoder_path}")
    
    # =========================================================================
    # Step 3: Train Deep Learning Model
    # =========================================================================
    print(f"\n[Step 3] Training {args.model_type} model...")
    
    # Model configuration
    model_config = AutoencoderConfig(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        latent_dim=args.latent_dim,
        dropout=0.2,
        use_batch_norm=True
    )
    
    # Training configuration
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        patience=10,
        device=args.device
    )
    
    # Split data
    n_samples = len(X)
    indices = torch.randperm(n_samples)
    train_size = int(0.8 * n_samples)
    
    X_train = X[indices[:train_size]]
    X_val = X[indices[train_size:]]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Create model
    if args.model_type == 'autoencoder':
        model = SecurityEventAutoencoder(model_config)
    elif args.model_type == 'vae':
        model = SecurityEventVAE(model_config)
    elif args.model_type == 'dec':
        n_clusters = args.n_clusters if args.n_clusters > 0 else 10
        model = DeepEmbeddingClustering(model_config, n_clusters=n_clusters)
    
    # Train
    trainer = SecurityEventTrainer(model, train_config)
    print(f"  Using device: {trainer.device}")
    
    if args.model_type == 'autoencoder':
        history = trainer.train_autoencoder(X_train, X_val)
    elif args.model_type == 'vae':
        history = trainer.train_vae(X_train, X_val)
    elif args.model_type == 'dec':
        history = trainer.train_dec(X_train)
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{args.model_type}_model.pt')
    trainer.save_model(model_path)
    
    # Plot training history
    history_plot = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=history_plot)
    
    # =========================================================================
    # Step 4: Extract Embeddings
    # =========================================================================
    print("\n[Step 4] Extracting embeddings...")
    
    embeddings = trainer.get_embeddings(X)
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Save embeddings
    embeddings_path = os.path.join(args.output_dir, 'embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f"  Embeddings saved to {embeddings_path}")
    
    # =========================================================================
    # Step 5: Find Optimal Clusters (if needed)
    # =========================================================================
    if args.n_clusters == 0 and args.clustering in ['kmeans', 'gmm']:
        print("\n[Step 5] Finding optimal number of clusters...")
        
        optimal_k, results = find_optimal_clusters(
            embeddings,
            min_clusters=2,
            max_clusters=20,
            method='silhouette'
        )
        
        print(f"  Optimal clusters: {optimal_k}")
        args.n_clusters = optimal_k
        
        # Plot elbow curve
        elbow_plot = os.path.join(args.output_dir, 'cluster_selection.png')
        plot_elbow_curve(results, save_path=elbow_plot)
    
    # =========================================================================
    # Step 6: Cluster Events
    # =========================================================================
    print(f"\n[Step 6] Clustering with {args.clustering}...")
    
    cluster_config = ClusteringConfig(
        method=args.clustering,
        n_clusters=args.n_clusters if args.n_clusters > 0 else 10,
        eps=0.5,
        min_samples=5,
        min_cluster_size=10
    )
    
    clusterer = SecurityEventClusterer(cluster_config)
    labels = clusterer.fit_predict(embeddings)
    
    print(f"  Number of clusters: {clusterer.n_clusters_}")
    print(f"  Cluster sizes: {clusterer.get_cluster_sizes()}")
    
    if clusterer.metrics_:
        print(f"  Metrics:")
        for key, value in clusterer.metrics_.items():
            if value is not None:
                print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
    
    # Save labels
    labels_path = os.path.join(args.output_dir, 'cluster_labels.npy')
    np.save(labels_path, labels)
    
    # =========================================================================
    # Step 7: Visualize Clusters
    # =========================================================================
    print("\n[Step 7] Creating visualizations...")
    
    # Reduce dimensions for visualization
    print("  Reducing dimensions with UMAP...")
    embeddings_2d = reduce_dimensions(embeddings, method='umap')
    
    # Plot clusters
    cluster_plot = os.path.join(args.output_dir, 'clusters_2d.png')
    plot_clusters(
        embeddings_2d,
        labels,
        title=f'Security Event Clusters ({args.clustering.upper()})',
        save_path=cluster_plot
    )
    
    # Plot distribution
    dist_plot = os.path.join(args.output_dir, 'cluster_distribution.png')
    plot_cluster_distribution(labels, save_path=dist_plot)
    
    # =========================================================================
    # Step 8: Analyze Clusters
    # =========================================================================
    print("\n[Step 8] Analyzing clusters...")
    
    analyzer = ClusterAnalyzer(labels)
    summaries = analyzer.get_cluster_summary(df)
    
    # Print top clusters
    print("\n  Top Clusters by Size:")
    for summary in sorted(summaries, key=lambda x: x['size'], reverse=True)[:5]:
        print(f"    Cluster {summary['cluster_id']}: {summary['size']} events ({summary['percentage']:.1f}%)")
        if 'top_subsystem' in summary:
            top_sub = list(summary['top_subsystem'].items())[:3]
            print(f"      Subsystems: {', '.join([f'{k}' for k, v in top_sub])}")
    
    # Identify anomalous clusters
    anomalous = analyzer.identify_anomalous_clusters(df)
    if anomalous:
        print(f"\n  Potentially Anomalous Clusters: {anomalous}")
    
    # Plot cluster analysis
    if 'subsystem' in df.columns:
        subsys_plot = os.path.join(args.output_dir, 'cluster_subsystems.png')
        plot_cluster_analysis(summaries, feature='subsystem', save_path=subsys_plot)
    
    if 'severity' in df.columns:
        severity_plot = os.path.join(args.output_dir, 'cluster_severity.png')
        plot_cluster_analysis(summaries, feature='severity', save_path=severity_plot)
    
    # Create text report
    report_path = os.path.join(args.output_dir, 'cluster_report.txt')
    create_cluster_report(summaries, clusterer.metrics_, report_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("CLUSTERING COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {args.output_dir}/")
    print("  - encoder.pkl (feature encoder)")
    print(f"  - {args.model_type}_model.pt (trained model)")
    print("  - embeddings.npy (learned embeddings)")
    print("  - cluster_labels.npy (cluster assignments)")
    print("  - training_history.png")
    print("  - clusters_2d.png")
    print("  - cluster_distribution.png")
    print("  - cluster_report.txt")
    
    print(f"\nTotal events: {len(df)}")
    print(f"Clusters found: {clusterer.n_clusters_}")
    if anomalous:
        print(f"Anomalous clusters: {len(anomalous)}")


if __name__ == "__main__":
    main()
