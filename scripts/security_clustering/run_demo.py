#!/usr/bin/env python3
"""
Demo Script - Security Event Clustering
Demonstrates the complete pipeline with synthetic data
"""

import os
import sys

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.parser import SecurityEventParser
from src.feature_encoder import FeatureEncoder, EncoderConfig
from src.models import SecurityEventVAE, AutoencoderConfig
from src.trainer import SecurityEventTrainer, TrainingConfig
from src.clustering import SecurityEventClusterer, ClusteringConfig, ClusterAnalyzer, find_optimal_clusters
from src.visualization import reduce_dimensions, plot_clusters, plot_cluster_distribution, plot_training_history
from src.data_generator import SecurityEventGenerator


def run_demo():
    """Run complete security event clustering demo"""
    
    print("=" * 70)
    print("SECURITY EVENT CLUSTERING - DEMO")
    print("Deep Learning Pipeline using PyTorch")
    print("=" * 70)
    
    # Configuration
    N_EVENTS = 3000
    LATENT_DIM = 32
    EPOCHS = 30
    BATCH_SIZE = 128
    OUTPUT_DIR = "demo_output"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. GENERATE SYNTHETIC DATA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Generating Synthetic Security Events")
    print("=" * 70)
    
    generator = SecurityEventGenerator(seed=42)
    
    # Custom distribution emphasizing attack patterns for interesting clusters
    pattern_distribution = {
        'sql_injection': 0.08,
        'bruteforce': 0.10,
        'ddos': 0.05,
        'malware': 0.07,
        'normal': 0.35,
        'policy_violation': 0.15,
        'port_scan': 0.05,
        'vpn_activity': 0.15,
    }
    
    events_raw = generator.generate_dataset(
        n_events=N_EVENTS,
        pattern_distribution=pattern_distribution
    )
    
    # Save generated data
    data_path = os.path.join(OUTPUT_DIR, 'security_events.log')
    generator.save_dataset(events_raw, data_path)
    
    print(f"\nGenerated {N_EVENTS} events")
    print(f"Sample events:")
    for i, event in enumerate(events_raw[:3]):
        print(f"  [{i+1}] {event[:100]}...")
    
    # -------------------------------------------------------------------------
    # 2. PARSE AND PROCESS EVENTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Parsing Security Events")
    print("=" * 70)
    
    parser = SecurityEventParser()
    events = parser.parse_lines(events_raw)
    df = parser.events_to_dataframe(events)
    
    print(f"\nDataset Statistics:")
    print(f"  Total events: {len(df)}")
    print(f"  Subsystems: {df['subsystem'].nunique()} unique")
    print(f"  Subsystem distribution:")
    for subsys, count in df['subsystem'].value_counts().head(5).items():
        print(f"    - {subsys}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n  Severity distribution:")
    for sev, count in df['severity'].value_counts().items():
        print(f"    - {sev}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n  Top destination ports:")
    for port, count in df['dest_port'].value_counts().head(5).items():
        print(f"    - Port {port}: {count}")
    
    # -------------------------------------------------------------------------
    # 3. FEATURE ENCODING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Encoding Features for Deep Learning")
    print("=" * 70)
    
    encoder_config = EncoderConfig(
        max_vocab_size=5000,
        min_freq=2,
        max_content_tokens=50
    )
    encoder = FeatureEncoder(encoder_config)
    encoder.fit(df)
    
    X = encoder.transform(df)
    input_dim = X.shape[1]
    
    print(f"\nFeature encoding complete:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Tensor shape: {X.shape}")
    print(f"  Feature groups:")
    print(f"    - Categorical vocabularies: {len(encoder.vocabularies)}")
    print(f"    - Numerical features: {len(encoder.numerical_stats)}")
    print(f"    - Content vocabulary size: {len(encoder.content_vocab)}")
    
    # Save encoder
    encoder.save(os.path.join(OUTPUT_DIR, 'encoder.pkl'))
    
    # -------------------------------------------------------------------------
    # 4. TRAIN VAE MODEL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Training Variational Autoencoder")
    print("=" * 70)
    
    # Model config
    model_config = AutoencoderConfig(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        latent_dim=LATENT_DIM,
        dropout=0.2,
        use_batch_norm=True
    )
    
    # Training config
    train_config = TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        patience=10,
        kl_weight=0.001,
        kl_annealing=True
    )
    
    # Split data
    n_samples = len(X)
    indices = torch.randperm(n_samples)
    train_size = int(0.8 * n_samples)
    
    X_train = X[indices[:train_size]]
    X_val = X[indices[train_size:]]
    
    print(f"\nModel Architecture:")
    print(f"  Input: {input_dim} -> Hidden: {model_config.hidden_dims} -> Latent: {LATENT_DIM}")
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Create and train model
    model = SecurityEventVAE(model_config)
    trainer = SecurityEventTrainer(model, train_config)
    
    print(f"\nDevice: {trainer.device}")
    print("\nTraining progress:")
    
    history = trainer.train_vae(X_train, X_val)
    
    # Save model
    trainer.save_model(os.path.join(OUTPUT_DIR, 'vae_model.pt'))
    
    # Plot training history
    plot_training_history(
        history,
        title="VAE Training History",
        save_path=os.path.join(OUTPUT_DIR, 'training_history.png')
    )
    
    # -------------------------------------------------------------------------
    # 5. EXTRACT EMBEDDINGS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Extracting Latent Embeddings")
    print("=" * 70)
    
    embeddings = trainer.get_embeddings(X)
    
    print(f"\nEmbedding Statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    
    # Save embeddings
    np.save(os.path.join(OUTPUT_DIR, 'embeddings.npy'), embeddings)
    
    # -------------------------------------------------------------------------
    # 6. FIND OPTIMAL CLUSTERS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Finding Optimal Number of Clusters")
    print("=" * 70)
    
    optimal_k, results = find_optimal_clusters(
        embeddings,
        min_clusters=3,
        max_clusters=15,
        method='silhouette'
    )
    
    print(f"\nCluster Selection Analysis:")
    print(f"  {'k':<5} {'Inertia':<12} {'Silhouette':<12} {'Calinski':<12}")
    print(f"  {'-'*41}")
    for r in results:
        sil = f"{r['silhouette']:.4f}" if r['silhouette'] else "N/A"
        cal = f"{r['calinski']:.1f}" if r['calinski'] else "N/A"
        print(f"  {r['k']:<5} {r['inertia']:<12.1f} {sil:<12} {cal:<12}")
    
    print(f"\n  Optimal clusters (by silhouette): {optimal_k}")
    
    # -------------------------------------------------------------------------
    # 7. CLUSTER EVENTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: Clustering Security Events")
    print("=" * 70)
    
    cluster_config = ClusteringConfig(
        method='kmeans',
        n_clusters=optimal_k
    )
    
    clusterer = SecurityEventClusterer(cluster_config)
    labels = clusterer.fit_predict(embeddings)
    
    print(f"\nClustering Results:")
    print(f"  Algorithm: K-Means")
    print(f"  Number of clusters: {clusterer.n_clusters_}")
    
    print(f"\nCluster Sizes:")
    sizes = clusterer.get_cluster_sizes()
    for cluster_id, size in sorted(sizes.items(), key=lambda x: x[1], reverse=True):
        pct = size / len(labels) * 100
        print(f"  Cluster {cluster_id}: {size:>5} events ({pct:>5.1f}%)")
    
    print(f"\nClustering Metrics:")
    for metric, value in clusterer.metrics_.items():
        if value is not None and isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    # Save labels
    np.save(os.path.join(OUTPUT_DIR, 'cluster_labels.npy'), labels)
    
    # -------------------------------------------------------------------------
    # 8. VISUALIZE RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8: Visualizing Clusters")
    print("=" * 70)
    
    print("\nReducing dimensions with UMAP for visualization...")
    embeddings_2d = reduce_dimensions(embeddings, method='umap')
    
    # Create cluster visualization
    plot_clusters(
        embeddings_2d,
        labels,
        title=f'Security Event Clusters (K={optimal_k})',
        save_path=os.path.join(OUTPUT_DIR, 'clusters_2d.png')
    )
    
    # Create distribution plot
    plot_cluster_distribution(
        labels,
        title='Security Event Cluster Distribution',
        save_path=os.path.join(OUTPUT_DIR, 'cluster_distribution.png')
    )
    
    print(f"Visualizations saved to {OUTPUT_DIR}/")
    
    # -------------------------------------------------------------------------
    # 9. ANALYZE CLUSTERS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 9: Analyzing Cluster Characteristics")
    print("=" * 70)
    
    analyzer = ClusterAnalyzer(labels)
    summaries = analyzer.get_cluster_summary(df)
    
    print("\nCluster Analysis Summary:")
    for summary in sorted(summaries, key=lambda x: x['size'], reverse=True):
        print(f"\n{'='*50}")
        print(f"CLUSTER {summary['cluster_id']} ({summary['size']} events, {summary['percentage']:.1f}%)")
        print(f"{'='*50}")
        
        if 'top_subsystem' in summary:
            print(f"  Top Subsystems:")
            for subsys, count in list(summary['top_subsystem'].items())[:3]:
                print(f"    - {subsys}: {count}")
        
        if 'top_severity' in summary:
            print(f"  Severity Distribution:")
            for sev, count in list(summary['top_severity'].items())[:3]:
                print(f"    - {sev}: {count}")
        
        if 'top_dest_port' in summary:
            print(f"  Top Ports:")
            for port, count in list(summary['top_dest_port'].items())[:3]:
                print(f"    - Port {int(port)}: {count}")
        
        if 'top_content_words' in summary:
            words = list(summary['top_content_words'].keys())[:5]
            print(f"  Key Terms: {', '.join(words)}")
    
    # Identify anomalous clusters
    anomalous = analyzer.identify_anomalous_clusters(df)
    if anomalous:
        print(f"\n⚠️  Potentially Anomalous Clusters: {anomalous}")
        print("   (Small size or high severity concentration)")
    
    # -------------------------------------------------------------------------
    # 10. SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 Results Summary:")
    print(f"   - Processed {N_EVENTS} security events")
    print(f"   - Extracted {LATENT_DIM}-dimensional embeddings using VAE")
    print(f"   - Identified {optimal_k} distinct clusters")
    
    print(f"\n📁 Output Files:")
    print(f"   - {OUTPUT_DIR}/security_events.log (raw events)")
    print(f"   - {OUTPUT_DIR}/encoder.pkl (feature encoder)")
    print(f"   - {OUTPUT_DIR}/vae_model.pt (trained model)")
    print(f"   - {OUTPUT_DIR}/embeddings.npy (latent embeddings)")
    print(f"   - {OUTPUT_DIR}/cluster_labels.npy (cluster assignments)")
    print(f"   - {OUTPUT_DIR}/training_history.png")
    print(f"   - {OUTPUT_DIR}/clusters_2d.png")
    print(f"   - {OUTPUT_DIR}/cluster_distribution.png")
    
    print(f"\n🔍 Security Insights:")
    for summary in sorted(summaries, key=lambda x: x['size'], reverse=True)[:3]:
        subsys = list(summary.get('top_subsystem', {}).keys())
        if subsys:
            print(f"   - Cluster {summary['cluster_id']}: Primarily {subsys[0]} events ({summary['percentage']:.0f}%)")
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
