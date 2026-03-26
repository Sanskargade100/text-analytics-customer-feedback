import pandas as pd
import matplotlib.pyplot as plt
import os

# Create results/plots directory
os.makedirs('results/plots', exist_ok=True)

def plot_metrics(lang, vector_name):
    file_path = f'results/clustering/metrics_{lang}_{vector_name}.csv'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # SSE Plot
    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia (SSE)', color=color)
    ax1.plot(df['k'], df['sse'], marker='o', color=color, label='SSE')
    ax1.tick_params(axis='y', labelcolor=color)

    # Silhouette Plot
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(df['k'], df['silhouette'], marker='s', color=color, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Clustering Stability Analysis ({lang.upper()} - {vector_name.upper()})')
    fig.tight_layout()
    plt.savefig(f'results/plots/stability_{lang}_{vector_name}.png')
    plt.close()

# Plot for all combinations
print("Generating stability plots...")
plot_metrics('en', 'tfidf')
plot_metrics('en', 'emb')
plot_metrics('de', 'tfidf')
plot_metrics('de', 'emb')

# Generate Stability Conclusion
with open('results/clustering/stability_conclusion.md', 'w', encoding='utf-8') as f:
    f.write("# Clustering Stability & k-Selection Analysis\n\n")
    f.write("Plots generated successfully. The visual analysis of SSE (elbow method) and Silhouette scores will guide the final parameter selections (vectorization choice and k-value), which are discussed in detail in the main pipeline report.\n")

print("Plots and conclusion generated in results/plots/ and results/clustering/.")
