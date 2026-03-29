#!/usr/bin/env python3
"""
Deep Learning Clustering Demo - Security Event Analysis
Demonstrates DEC, IDEC, VaDE, and DCN clustering methods

These are TRUE deep learning clustering methods where:
- The neural network learns to cluster directly (not just feature extraction)
- Clustering objective is part of the loss function
- No traditional ML algorithms (K-Means, DBSCAN) are used for final clustering
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
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from src.parser import SecurityEventParser
from src.feature_encoder import FeatureEncoder, EncoderConfig
from src.data_generator import SecurityEventGenerator
from src.deep_clustering import (
    DeepClusteringConfig,
    DEC,
    IDEC,
    VaDE,
    DCN,
    DeepClusteringTrainer
)
from src.visualization import reduce_dimensions, plot_clusters


def evaluate_clustering(true_labels, pred_labels):
    """Evaluate clustering quality with ground truth"""
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return nmi, ari


def run_deep_clustering_demo():
    """Run deep learning clustering demo"""
    
    print("=" * 70)
    print("DEEP LEARNING CLUSTERING - Security Events")
    print("Methods: DEC, IDEC, VaDE, DCN")
    print("=" * 70)
    
    # Configuration
    N_EVENTS = 5000
    LATENT_DIM = 32
    N_CLUSTERS = 8
    PRETRAIN_EPOCHS = 30
    CLUSTER_EPOCHS = 50
    BATCH_SIZE = 256
    OUTPUT_DIR = "deep_clustering_output"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. GENERATE DATA WITH KNOWN PATTERNS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Generating Security Events with Known Attack Patterns")
    print("=" * 70)
    
    generator = SecurityEventGenerator(seed=42)
    
    # Define 8 distinct attack patterns for clustering
    pattern_distribution = {
        'sql_injection': 0.12,    # Cluster 0
        'bruteforce': 0.12,       # Cluster 1
        'ddos': 0.10,             # Cluster 2
        'malware': 0.12,          # Cluster 3
        'normal': 0.18,           # Cluster 4
        'policy_violation': 0.12, # Cluster 5
        'port_scan': 0.12,        # Cluster 6
        'vpn_activity': 0.12,     # Cluster 7
    }
    
    events_raw = generator.generate_dataset(
        n_events=N_EVENTS,
        pattern_distribution=pattern_distribution
    )
    
    # Get ground truth labels from generator
    ground_truth = generator.get_labels(events_raw)
    
    print(f"Generated {N_EVENTS} events with {N_CLUSTERS} attack patterns")
    print(f"Pattern distribution:")
    for pattern, pct in pattern_distribution.items():
        count = int(N_EVENTS * pct)
        print(f"  - {pattern}: ~{count} events ({pct*100:.0f}%)")
    
    # -------------------------------------------------------------------------
    # 2. PARSE AND ENCODE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Parsing and Encoding Events")
    print("=" * 70)
    
    parser = SecurityEventParser()
    events = parser.parse_lines(events_raw)
    df = parser.events_to_dataframe(events)
    
    encoder_config = EncoderConfig(max_vocab_size=5000, min_freq=2)
    encoder = FeatureEncoder(encoder_config)
    encoder.fit(df)
    
    X = encoder.transform(df)
    input_dim = X.shape[1]
    
    print(f"Input dimension: {input_dim}")
    print(f"Data shape: {X.shape}")
    
    # -------------------------------------------------------------------------
    # 3. DEEP CLUSTERING MODELS
    # -------------------------------------------------------------------------
    
    # Store results for comparison
    results = {}
    
    # Define model configurations
    config = DeepClusteringConfig(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        latent_dim=LATENT_DIM,
        n_clusters=N_CLUSTERS,
        dropout=0.2,
        use_batch_norm=True
    )
    
    # -------------------------------------------------------------------------
    # Method 1: DEC (Deep Embedded Clustering)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("METHOD 1: DEC (Deep Embedded Clustering)")
    print("=" * 70)
    print("""
    DEC learns clustering by:
    1. Pretraining an autoencoder for feature representation
    2. Initializing cluster centers with K-means on latent space
    3. Fine-tuning with KL divergence between soft assignments (Q) 
       and an auxiliary target distribution (P)
    """)
    
    dec_model = DEC(config)
    dec_trainer = DeepClusteringTrainer(dec_model, learning_rate=1e-3)
    
    # Pretrain autoencoder
    dec_trainer.pretrain_autoencoder(X, epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE)
    
    # Train clustering
    dec_trainer.train_dec(X, epochs=CLUSTER_EPOCHS, batch_size=BATCH_SIZE)
    
    # Get results
    dec_labels = dec_trainer.get_cluster_assignments(X)
    dec_embeddings = dec_trainer.get_embeddings(X)
    
    nmi, ari = evaluate_clustering(ground_truth, dec_labels)
    results['DEC'] = {'NMI': nmi, 'ARI': ari, 'labels': dec_labels, 'embeddings': dec_embeddings}
    
    print(f"\nDEC Results:")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    
    # Save model
    dec_trainer.save_model(os.path.join(OUTPUT_DIR, 'dec_model.pt'))
    
    # -------------------------------------------------------------------------
    # Method 2: IDEC (Improved DEC)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("METHOD 2: IDEC (Improved Deep Embedded Clustering)")
    print("=" * 70)
    print("""
    IDEC improves DEC by:
    - Adding reconstruction loss to preserve local structure
    - Loss = KL_clustering + gamma * reconstruction_loss
    - Better preserves data manifold structure
    """)
    
    idec_model = IDEC(config, gamma=0.1)
    idec_trainer = DeepClusteringTrainer(idec_model, learning_rate=1e-3)
    
    # Pretrain autoencoder
    idec_trainer.pretrain_autoencoder(X, epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE)
    
    # Train clustering
    idec_trainer.train_dec(X, epochs=CLUSTER_EPOCHS, batch_size=BATCH_SIZE)
    
    # Get results
    idec_labels = idec_trainer.get_cluster_assignments(X)
    idec_embeddings = idec_trainer.get_embeddings(X)
    
    nmi, ari = evaluate_clustering(ground_truth, idec_labels)
    results['IDEC'] = {'NMI': nmi, 'ARI': ari, 'labels': idec_labels, 'embeddings': idec_embeddings}
    
    print(f"\nIDEC Results:")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    
    # Save model
    idec_trainer.save_model(os.path.join(OUTPUT_DIR, 'idec_model.pt'))
    
    # -------------------------------------------------------------------------
    # Method 3: VaDE (Variational Deep Embedding)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("METHOD 3: VaDE (Variational Deep Embedding)")
    print("=" * 70)
    print("""
    VaDE uses:
    - Variational Autoencoder (VAE) architecture
    - Gaussian Mixture Model (GMM) prior in latent space
    - Each cluster is a Gaussian component
    - Learns both representation and clustering via ELBO
    """)
    
    vade_model = VaDE(config)
    vade_trainer = DeepClusteringTrainer(vade_model, learning_rate=1e-3)
    
    # Pretrain VAE
    vade_trainer.pretrain_autoencoder(X, epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE)
    
    # Train VaDE clustering
    vade_trainer.train_vade(X, epochs=CLUSTER_EPOCHS, batch_size=BATCH_SIZE)
    
    # Get results
    vade_labels = vade_trainer.get_cluster_assignments(X)
    vade_embeddings = vade_trainer.get_embeddings(X)
    
    nmi, ari = evaluate_clustering(ground_truth, vade_labels)
    results['VaDE'] = {'NMI': nmi, 'ARI': ari, 'labels': vade_labels, 'embeddings': vade_embeddings}
    
    print(f"\nVaDE Results:")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    
    # Save model
    vade_trainer.save_model(os.path.join(OUTPUT_DIR, 'vade_model.pt'))
    
    # -------------------------------------------------------------------------
    # Method 4: DCN (Deep Clustering Network)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("METHOD 4: DCN (Deep Clustering Network)")
    print("=" * 70)
    print("""
    DCN jointly optimizes:
    - Reconstruction loss (autoencoder)
    - K-means loss (distance to nearest cluster center)
    - Loss = reconstruction + lambda * min_distance_to_center
    """)
    
    dcn_model = DCN(config, lambda_kmeans=0.1)
    dcn_trainer = DeepClusteringTrainer(dcn_model, learning_rate=1e-3)
    
    # Pretrain autoencoder
    dcn_trainer.pretrain_autoencoder(X, epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE)
    
    # Train DCN clustering
    dcn_trainer.train_dcn(X, epochs=CLUSTER_EPOCHS, batch_size=BATCH_SIZE)
    
    # Get results
    dcn_labels = dcn_trainer.get_cluster_assignments(X)
    dcn_embeddings = dcn_trainer.get_embeddings(X)
    
    nmi, ari = evaluate_clustering(ground_truth, dcn_labels)
    results['DCN'] = {'NMI': nmi, 'ARI': ari, 'labels': dcn_labels, 'embeddings': dcn_embeddings}
    
    print(f"\nDCN Results:")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    
    # Save model
    dcn_trainer.save_model(os.path.join(OUTPUT_DIR, 'dcn_model.pt'))
    
    # -------------------------------------------------------------------------
    # 4. COMPARISON AND VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON: All Deep Clustering Methods")
    print("=" * 70)
    
    print(f"\n{'Method':<10} {'NMI':>10} {'ARI':>10}")
    print("-" * 32)
    for method, res in sorted(results.items(), key=lambda x: x[1]['NMI'], reverse=True):
        print(f"{method:<10} {res['NMI']:>10.4f} {res['ARI']:>10.4f}")
    
    best_method = max(results.items(), key=lambda x: x[1]['NMI'])
    print(f"\nBest Method: {best_method[0]} (NMI: {best_method[1]['NMI']:.4f})")
    
    # -------------------------------------------------------------------------
    # 5. VISUALIZE BEST RESULT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Visualizing Cluster Results")
    print("=" * 70)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ground truth
    gt_embeddings = reduce_dimensions(results['DEC']['embeddings'], method='umap')
    ax = axes[0, 0]
    scatter = ax.scatter(gt_embeddings[:, 0], gt_embeddings[:, 1], 
                        c=ground_truth, cmap='tab10', s=5, alpha=0.6)
    ax.set_title('Ground Truth Patterns', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # DEC
    ax = axes[0, 1]
    dec_2d = reduce_dimensions(results['DEC']['embeddings'], method='umap')
    ax.scatter(dec_2d[:, 0], dec_2d[:, 1], 
              c=results['DEC']['labels'], cmap='tab10', s=5, alpha=0.6)
    ax.set_title(f"DEC (NMI: {results['DEC']['NMI']:.3f})", fontsize=14)
    ax.set_xlabel('UMAP 1')
    
    # IDEC
    ax = axes[0, 2]
    idec_2d = reduce_dimensions(results['IDEC']['embeddings'], method='umap')
    ax.scatter(idec_2d[:, 0], idec_2d[:, 1], 
              c=results['IDEC']['labels'], cmap='tab10', s=5, alpha=0.6)
    ax.set_title(f"IDEC (NMI: {results['IDEC']['NMI']:.3f})", fontsize=14)
    ax.set_xlabel('UMAP 1')
    
    # VaDE
    ax = axes[1, 0]
    vade_2d = reduce_dimensions(results['VaDE']['embeddings'], method='umap')
    ax.scatter(vade_2d[:, 0], vade_2d[:, 1], 
              c=results['VaDE']['labels'], cmap='tab10', s=5, alpha=0.6)
    ax.set_title(f"VaDE (NMI: {results['VaDE']['NMI']:.3f})", fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # DCN
    ax = axes[1, 1]
    dcn_2d = reduce_dimensions(results['DCN']['embeddings'], method='umap')
    ax.scatter(dcn_2d[:, 0], dcn_2d[:, 1], 
              c=results['DCN']['labels'], cmap='tab10', s=5, alpha=0.6)
    ax.set_title(f"DCN (NMI: {results['DCN']['NMI']:.3f})", fontsize=14)
    ax.set_xlabel('UMAP 1')
    
    # Comparison bar chart
    ax = axes[1, 2]
    methods = list(results.keys())
    nmi_scores = [results[m]['NMI'] for m in methods]
    ari_scores = [results[m]['ARI'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, nmi_scores, width, label='NMI', color='steelblue')
    bars2 = ax.bar(x + width/2, ari_scores, width, label='ARI', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Clustering Quality Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'deep_clustering_comparison.png'), dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {OUTPUT_DIR}/deep_clustering_comparison.png")
    
    # -------------------------------------------------------------------------
    # 6. SECURITY ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECURITY INSIGHTS from Deep Clustering")
    print("=" * 70)
    
    # Use best method's labels
    best_labels = best_method[1]['labels']
    
    print(f"\nUsing {best_method[0]} results for security analysis:")
    print(f"\nCluster Distribution:")
    
    unique, counts = np.unique(best_labels, return_counts=True)
    for cluster_id, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
        pct = count / len(best_labels) * 100
        
        # Analyze cluster composition
        cluster_mask = best_labels == cluster_id
        cluster_df = df[cluster_mask]
        
        # Get dominant subsystem
        top_subsys = cluster_df['subsystem'].mode().iloc[0] if len(cluster_df) > 0 else 'unknown'
        
        # Get dominant severity
        top_severity = cluster_df['severity'].mode().iloc[0] if len(cluster_df) > 0 else 'unknown'
        
        # Get top port
        top_port = cluster_df['dest_port'].mode().iloc[0] if len(cluster_df['dest_port'].dropna()) > 0 else 'N/A'
        
        print(f"\n  Cluster {cluster_id}: {count} events ({pct:.1f}%)")
        print(f"    - Dominant Subsystem: {top_subsys}")
        print(f"    - Primary Severity: {top_severity}")
        print(f"    - Common Port: {top_port}")
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    
    print(f"""
Summary:
--------
This demo compared 4 deep learning clustering methods:

1. DEC (Deep Embedded Clustering)
   - Uses KL divergence between soft assignments and target distribution
   - Most widely used deep clustering method

2. IDEC (Improved DEC)
   - Adds reconstruction loss to preserve local structure
   - Better for maintaining data manifold

3. VaDE (Variational Deep Embedding)
   - VAE with GMM prior in latent space
   - Probabilistic cluster assignments

4. DCN (Deep Clustering Network)
   - Joint optimization of reconstruction + K-means
   - K-means friendly representation learning

Key Differences from Traditional ML:
------------------------------------
- Clustering is LEARNED end-to-end (not a separate step)
- No sklearn K-means/DBSCAN for final clustering
- Neural network directly outputs cluster assignments
- Representation and clustering are jointly optimized

Output Files:
-------------
- {OUTPUT_DIR}/dec_model.pt
- {OUTPUT_DIR}/idec_model.pt
- {OUTPUT_DIR}/vade_model.pt
- {OUTPUT_DIR}/dcn_model.pt
- {OUTPUT_DIR}/deep_clustering_comparison.png

Best Method: {best_method[0]} (NMI: {best_method[1]['NMI']:.4f}, ARI: {best_method[1]['ARI']:.4f})
""")


if __name__ == "__main__":
    run_deep_clustering_demo()
