"""
Inference Script for Security Event Clustering
Load trained model and cluster new events
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser import SecurityEventParser
from src.feature_encoder import FeatureEncoder
from src.models import SecurityEventAutoencoder, SecurityEventVAE, AutoencoderConfig
from src.clustering import SecurityEventClusterer, ClusteringConfig, ClusterAnalyzer


def load_model(model_path: str, model_type: str, input_dim: int, latent_dim: int = 32, device: str = 'cpu'):
    """Load trained model"""
    config = AutoencoderConfig(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        latent_dim=latent_dim,
        dropout=0.2,
        use_batch_norm=True
    )
    
    if model_type == 'autoencoder':
        model = SecurityEventAutoencoder(config)
    elif model_type == 'vae':
        model = SecurityEventVAE(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Inference for Security Event Clustering')
    parser.add_argument('--input', '-i', required=True, help='Input file with new events')
    parser.add_argument('--model-dir', required=True, help='Directory with trained model')
    parser.add_argument('--model-type', default='vae', choices=['autoencoder', 'vae'])
    parser.add_argument('--output', '-o', default='inference_results', help='Output directory')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading encoder...")
    encoder_path = os.path.join(args.model_dir, 'encoder.pkl')
    encoder = FeatureEncoder.load(encoder_path)
    
    print("Loading model...")
    model_path = os.path.join(args.model_dir, f'{args.model_type}_model.pt')
    
    # Get input dimension from encoder
    input_dim = encoder.get_total_dim()
    model = load_model(model_path, args.model_type, input_dim, device=args.device)
    
    print("Parsing events...")
    event_parser = SecurityEventParser()
    with open(args.input, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    events = event_parser.parse_lines(lines)
    df = event_parser.events_to_dataframe(events)
    
    print(f"Processing {len(df)} events...")
    
    # Encode features
    X = encoder.transform(df)
    
    # Get embeddings
    with torch.no_grad():
        X_tensor = X.to(args.device)
        embeddings = model.encode(X_tensor).cpu().numpy()
    
    print("Clustering...")
    # Load reference cluster centers if available
    ref_embeddings_path = os.path.join(args.model_dir, 'embeddings.npy')
    ref_labels_path = os.path.join(args.model_dir, 'cluster_labels.npy')
    
    if os.path.exists(ref_embeddings_path) and os.path.exists(ref_labels_path):
        ref_embeddings = np.load(ref_embeddings_path)
        ref_labels = np.load(ref_labels_path)
        
        # Use same clustering configuration
        from sklearn.cluster import KMeans
        n_clusters = len(set(ref_labels)) - (1 if -1 in ref_labels else 0)
        
        # Fit KMeans on reference data and predict on new data
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(ref_embeddings[ref_labels != -1])
        labels = kmeans.predict(embeddings)
    else:
        # Fresh clustering
        config = ClusteringConfig(method='kmeans', n_clusters=10)
        clusterer = SecurityEventClusterer(config)
        labels = clusterer.fit_predict(embeddings)
    
    # Analyze
    analyzer = ClusterAnalyzer(labels)
    summaries = analyzer.get_cluster_summary(df)
    
    # Save results
    np.save(os.path.join(args.output, 'labels.npy'), labels)
    np.save(os.path.join(args.output, 'embeddings.npy'), embeddings)
    
    # Add cluster labels to dataframe and save
    df['cluster'] = labels
    df.to_csv(os.path.join(args.output, 'clustered_events.csv'), index=False)
    
    print(f"\nResults saved to {args.output}/")
    print(f"Cluster distribution:")
    for cluster_id in sorted(set(labels)):
        count = (labels == cluster_id).sum()
        print(f"  Cluster {cluster_id}: {count} events")


if __name__ == "__main__":
    main()
