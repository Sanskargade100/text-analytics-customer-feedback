import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# Create results/interpretation directory
os.makedirs('results/interpretation', exist_ok=True)

def get_ctfidf(texts, labels, n_clusters, lang='en'):
    """Calculate c-TF-IDF keywords for each cluster."""
    # Group texts by cluster
    documents = pd.DataFrame({'text': [str(t) if pd.notnull(t) else "" for t in texts], 'cluster': labels})
    documents_per_cluster = documents.groupby(['cluster'], as_index=False).agg({'text': ' '.join})
    
    # Calculate count of words per cluster
    if lang == 'en':
        stop_words_list = 'english'
    else:
        try:
            from spacy.lang.de.stop_words import STOP_WORDS
            stop_words_list = list(STOP_WORDS)
        except ImportError:
            stop_words_list = None
            
    count_vectorizer = CountVectorizer(stop_words=stop_words_list, min_df=2)
    count = count_vectorizer.fit_transform(documents_per_cluster.text)
    words = count_vectorizer.get_feature_names_out()
    
    # Calculate c-TF-IDF
    t = count.toarray()
    w = t.sum(axis=1)
    tf = t / w.reshape(-1, 1)
    
    idf = np.log(1 + (w.mean() / (t.sum(axis=0) + 1e-6)))
    ctfidf = tf * idf
    
    # Get top words
    top_words = {}
    for i in range(n_clusters):
        top_indices = ctfidf[i].argsort()[-10:][::-1]
        top_words[i] = [words[idx] for idx in top_indices]
    
    return top_words

def interpret_clusters(lang, df, vectors, k=15):
    print(f"Interpreting clusters for {lang}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    df['cluster'] = labels
    
    # Get c-TF-IDF words
    top_words = get_ctfidf(df['processed_text'].tolist(), labels, k, lang)
    
    # Get representative tickets (closest to centroid)
    dist = kmeans.transform(vectors) # Distance to each centroid
    
    report = []

    def safe_mode(series):
        mode = series.dropna().mode()
        return mode.iloc[0] if not mode.empty else "N/A"

    for i in range(k):
        # Top 3 representative indices within the cluster
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            rep_tickets = []
        else:
            cluster_dist = dist[cluster_indices, i]
            closest_idx_within_cluster = cluster_indices[cluster_dist.argsort()[:3]]
            rep_tickets = df.iloc[closest_idx_within_cluster][['subject', 'queue', 'tag_1']].values.tolist()
        
        # Most frequent queue/tag in cluster
        cluster_df = df[df['cluster'] == i]
        top_queue = safe_mode(cluster_df['queue'])
        top_tag = safe_mode(cluster_df['tag_1'])
        
        cluster_info = {
            'cluster_id': i,
            'size': int((labels == i).sum()),
            'top_words': ", ".join(top_words[i]),
            'top_queue': top_queue,
            'top_tag': top_tag,
            'rep_tickets': rep_tickets
        }
        report.append(cluster_info)
    
    # Save report
    rep_df = pd.DataFrame(report)
    rep_df.to_csv(f'results/interpretation/cluster_report_{lang}.csv', index=False)
    
    # Save df with labels
    df.to_csv(f'results/interpretation/df_{lang}_labeled.csv', index=False)
    
    return rep_df

# Load data and vectors
print("Loading data and vectors...")
df_en = pd.read_csv('results/processed/df_en_preprocessed.csv')
df_de = pd.read_csv('results/processed/df_de_preprocessed.csv')

with open('results/processed/vectors_en.pkl', 'rb') as f:
    v_en = pickle.load(f)
with open('results/processed/vectors_de.pkl', 'rb') as f:
    v_de = pickle.load(f)

# Interpret
interpret_clusters('en', df_en, v_en['emb'], k=15)
interpret_clusters('de', df_de, v_de['emb'], k=15)

print("Interpretation complete.")
