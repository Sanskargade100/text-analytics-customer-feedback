import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Create results/clustering directory
os.makedirs('results/clustering', exist_ok=True)

def run_clustering_analysis(lang, vectors, vector_name):
    print(f"Running clustering for {lang} - {vector_name}...")
    
    # Range of k
    k_range = range(5, 26, 2)
    sse = []
    silhouette_avg = []
    
    # K-Means analysis
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(vectors)
        sse.append(kmeans.inertia_)
        
        # Silhouette score (sampled if too large for speed)
        if vectors.shape[0] > 10000:
            sample_idx = np.random.choice(vectors.shape[0], 10000, replace=False)
            score = silhouette_score(vectors[sample_idx], kmeans.labels_[sample_idx])
        else:
            score = silhouette_score(vectors, kmeans.labels_)
        silhouette_avg.append(score)
        print(f"k={k}, SSE={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
    
    # Save metrics
    metrics = pd.DataFrame({
        'k': k_range,
        'sse': sse,
        'silhouette': silhouette_avg
    })
    metrics.to_csv(f'results/clustering/metrics_{lang}_{vector_name}.csv', index=False)
    
    return metrics

# Load vectors
print("Loading vectors...")
with open('results/processed/vectors_en.pkl', 'rb') as f:
    v_en = pickle.load(f)
with open('results/processed/vectors_de.pkl', 'rb') as f:
    v_de = pickle.load(f)

# Run for English
run_clustering_analysis('en', v_en['tfidf'], 'tfidf')
run_clustering_analysis('en', v_en['emb'], 'emb')

# Run for German
run_clustering_analysis('de', v_de['tfidf'], 'tfidf')
run_clustering_analysis('de', v_de['emb'], 'emb')

# Run HAC for Embeddings (fixed k=15 as a representative sample)
print("Running HAC for k=15...")
hac_en = AgglomerativeClustering(n_clusters=15, linkage='ward')
hac_labels_en = hac_en.fit_predict(v_en['emb'])

hac_de = AgglomerativeClustering(n_clusters=15, linkage='ward')
hac_labels_de = hac_de.fit_predict(v_de['emb'])

# Save HAC labels
np.save('results/clustering/hac_labels_en.npy', hac_labels_en)
np.save('results/clustering/hac_labels_de.npy', hac_labels_de)

print("Clustering analysis complete.")
