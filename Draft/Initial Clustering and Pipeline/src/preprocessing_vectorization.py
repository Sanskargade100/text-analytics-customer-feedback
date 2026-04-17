import pandas as pd
import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pickle

# Create a results/processed directory if it doesn't exist
os.makedirs('results/processed', exist_ok=True)

# Load sampled data
df_en = pd.read_csv('results/df_en_sampled.csv')
df_de = pd.read_csv('results/df_de_sampled.csv')

# Preprocessing function
def preprocess(texts, nlp_model):
    processed_texts = []
    # Using pipe for efficiency
    for doc in nlp_model.pipe(texts, disable=['ner', 'parser']):
        # Lowercase, remove punct/stopwords/digits, lemmatize
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_digit and len(token.text) > 2]
        processed_texts.append(" ".join(tokens))
    return processed_texts

print("Starting preprocessing...")

# Load Spacy models
print("Loading Spacy EN...")
nlp_en = spacy.load('en_core_web_sm')
print("Loading Spacy DE...")
nlp_de = spacy.load('de_core_news_sm')

# Process English
print("Processing English texts...")
df_en['text'] = df_en['text'].fillna('').astype(str)
df_en['processed_text'] = preprocess(df_en['text'], nlp_en)

# Process German
print("Processing German texts...")
df_de['text'] = df_de['text'].fillna('').astype(str)
df_de['processed_text'] = preprocess(df_de['text'], nlp_de)

# Vectorization - TF-IDF
print("Vectorizing TF-IDF...")
tfidf_en = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_tfidf_en = tfidf_en.fit_transform(df_en['processed_text'])

tfidf_de = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_tfidf_de = tfidf_de.fit_transform(df_de['processed_text'])

# Vectorization - Embeddings
print("Vectorizing Embeddings...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X_emb_en = model.encode(df_en['text'].tolist(), show_progress_bar=True)
X_emb_de = model.encode(df_de['text'].tolist(), show_progress_bar=True)

# Save processed data and vectors
print("Saving processed data and vectors...")
df_en.to_csv('results/processed/df_en_preprocessed.csv', index=False)
df_de.to_csv('results/processed/df_de_preprocessed.csv', index=False)

with open('results/processed/vectors_en.pkl', 'wb') as f:
    pickle.dump({'tfidf': X_tfidf_en, 'emb': X_emb_en, 'tfidf_model': tfidf_en}, f)

with open('results/processed/vectors_de.pkl', 'wb') as f:
    pickle.dump({'tfidf': X_tfidf_de, 'emb': X_emb_de, 'tfidf_model': tfidf_de}, f)

print("Preprocessing and vectorization complete.")
