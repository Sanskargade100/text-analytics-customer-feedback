import pandas as pd
from datasets import load_dataset
import os

# Create a results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

print("Loading dataset...")
dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
df = pd.DataFrame(dataset['train'])

print(f"Total records: {len(df)}")

# Record raw missingness before filling text fields
missing_values = df.isnull().sum()

# Create text column
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['text'] = df['subject'] + "\n" + df['body']

# Split by language
df_en = df[df['language'] == 'en'].copy()
df_de = df[df['language'] == 'de'].copy()

print(f"English records: {len(df_en)}")
print(f"German records: {len(df_de)}")

# Sampling 5000 per language
if len(df_en) > 5000:
    df_en = df_en.sample(5000, random_state=42)
if len(df_de) > 5000:
    df_de = df_de.sample(5000, random_state=42)

print(f"Sampled English: {len(df_en)}")
print(f"Sampled German: {len(df_de)}")

# EDA Analysis
df['text_length'] = df['text'].astype(str).str.len()
duplicate_values = df.duplicated(subset=['subject', 'body']).sum()
priority_dist = df['priority'].value_counts()
type_dist = df['type'].value_counts()
text_len_desc = df['text_length'].describe()

# Save basic stats
with open('results/data_stats.txt', 'w', encoding='utf-8') as f:
    f.write("Task 4: Exploring Data Distributions\n")
    f.write("====================================\n\n")
    f.write(f"Total records: {len(df)}\n")
    f.write(f"Duplicate records (based on subject+body): {duplicate_values}\n\n")
    f.write(f"Missing Values per Column:\n{missing_values.to_string()}\n\n")
    f.write(f"Priority Distribution:\n{priority_dist.to_string()}\n\n")
    f.write(f"Type Distribution:\n{type_dist.to_string()}\n\n")
    f.write(f"Text Length (characters) Distribution:\n{text_len_desc.to_string()}\n\n")
    f.write(f"English sampled: {len(df_en)}\n")
    f.write(f"German sampled: {len(df_de)}\n")
    f.write("\nTop Queues (EN):\n" + df_en['queue'].value_counts().head(5).to_string() + "\n")
    f.write("\nTop Queues (DE):\n" + df_de['queue'].value_counts().head(5).to_string() + "\n")

# Save sampled data
df_en.to_csv('results/df_en_sampled.csv', index=False)
df_de.to_csv('results/df_de_sampled.csv', index=False)

print("\nData exploration and sampling complete.")
