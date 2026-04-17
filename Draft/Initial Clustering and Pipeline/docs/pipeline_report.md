# Task 4 Report: Sections 1 and 2.a Only

## Scope

This document covers only the parts I am responsible for in Task 4:

1. Initial clustering analysis of issue types using K-Means or HAC on TF-IDF vectors or embeddings, including cluster coherence, interpretability, and stability across k.
2. The design and implementation of the text analytics pipeline in Task 2.a, including data exploration, preprocessing, text representation, issue discovery, and issue ranking logic.

It does not evaluate the later group sections in full detail.

## 1. Clustering of Issue Types

### 1.1 Methods Compared

I compared two text representations:

- TF-IDF vectors with unigrams and bigrams
- Multilingual sentence embeddings from `paraphrase-multilingual-MiniLM-L12-v2`

I used K-Means as the main clustering method and also ran HAC on embeddings at `k = 15` as a secondary comparison.

### 1.2 Stability Across k

For K-Means, I tested odd values of `k` from 5 to 25 for both English and German subsets and recorded:

- SSE / inertia
- silhouette score

The metric files are:

- [metrics_en_tfidf.csv](C:/Users/Soda/Desktop/test2/results/clustering/metrics_en_tfidf.csv)
- [metrics_en_emb.csv](C:/Users/Soda/Desktop/test2/results/clustering/metrics_en_emb.csv)
- [metrics_de_tfidf.csv](C:/Users/Soda/Desktop/test2/results/clustering/metrics_de_tfidf.csv)
- [metrics_de_emb.csv](C:/Users/Soda/Desktop/test2/results/clustering/metrics_de_emb.csv)

The corresponding plots are:

- [stability_en_tfidf.png](C:/Users/Soda/Desktop/test2/results/plots/stability_en_tfidf.png)
- [stability_en_emb.png](C:/Users/Soda/Desktop/test2/results/plots/stability_en_emb.png)
- [stability_de_tfidf.png](C:/Users/Soda/Desktop/test2/results/plots/stability_de_tfidf.png)
- [stability_de_emb.png](C:/Users/Soda/Desktop/test2/results/plots/stability_de_emb.png)

### 1.3 Main Findings from Stability

The results show a clear difference between TF-IDF and embeddings.

- English TF-IDF produced very low silhouette scores across all tested `k` values, mostly around `0.00-0.04`.
- German TF-IDF was even weaker, with several negative silhouette scores, which suggests poor cluster separation.
- English embeddings were much stronger, with silhouette values around `0.20-0.24`.
- German embeddings were also stronger than TF-IDF, with silhouette values around `0.11-0.14`.

This means embeddings created more stable and better-separated issue clusters than TF-IDF in both languages.

I selected `k = 15` for the later interpretation stage because it offered a practical balance between:

- enough detail to split broad issue families into useful sub-groups
- acceptable stability on the embedding representation
- manageable interpretability for manual inspection

Although the absolute best silhouette value for English embeddings appears at lower `k`, `k = 15` still remains competitive and gives richer issue granularity. For German embeddings, the scores are flatter across the tested range, so choosing `k = 15` is a defensible compromise between stability and interpretability.

### 1.4 Cluster Coherence and Interpretability

To assess coherence, I reviewed c-TF-IDF keywords and representative tickets for each embedding-based K-Means cluster.

English examples from [cluster_report_en.csv](C:/Users/Soda/Desktop/test2/results/interpretation/cluster_report_en.csv):

- Cluster 4 is coherent around medical data security breaches.
- Cluster 9 is coherent around billing and subscription issues.
- Cluster 11 is coherent around crashes and compatibility failures.
- Cluster 14 is coherent around outages and peak-load performance issues.

German examples from [cluster_report_de.csv](C:/Users/Soda/Desktop/test2/results/interpretation/cluster_report_de.csv):

- Cluster 2 is coherent around VPN connectivity issues.
- Cluster 6 is coherent around billing and invoice errors.
- Cluster 7 is coherent around medical data protection.
- Cluster 14 is coherent around data analysis tool bugs and failures.

There are also noisy clusters, which is important to acknowledge.

- English cluster 2 contains many rare tokens and weakly related terms, so it is less interpretable.
- German cluster 1 is largely noise and tokenization artifacts.
- Some German clusters still contain mixed English terms, which slightly reduces coherence.

Overall, the embedding-based clusters are interpretable enough for issue discovery, but not every cluster is equally clean. This is a realistic outcome for customer-support text.

### 1.5 HAC Comparison

I also ran HAC with Ward linkage on embeddings at `k = 15` and saved the labels in:

- [hac_labels_en.npy](C:/Users/Soda/Desktop/test2/results/clustering/hac_labels_en.npy)
- [hac_labels_de.npy](C:/Users/Soda/Desktop/test2/results/clustering/hac_labels_de.npy)

In this pipeline, HAC is included as a comparison method to satisfy the task requirement, while K-Means remains the main working model because it scales more easily, supports repeated stability analysis across many `k` values, and integrates naturally with representative-ticket inspection.

## 2. Pipeline Design and Implementation (Task 2.a)

### 2.1 Exploring Data Distributions

The first stage was exploratory data analysis on the raw ticket dataset. This stage is implemented in [data_exploration.py](C:/Users/Soda/Desktop/test2/data_exploration.py).

The output summary is stored in [data_stats.txt](C:/Users/Soda/Desktop/test2/results/data_stats.txt).

The EDA covers:

- total number of records
- missing values by column
- duplicate records based on `subject + body`
- priority distribution
- type distribution
- text length distribution
- top queue distribution for English and German samples

The current summary shows that the dataset is large enough for clustering, contains non-trivial missingness in fields such as `subject`, `type`, and tags, and includes both English and German records in sufficient quantities for separate analysis.

### 2.2 Language Split and Sampling

I split the corpus into English and German subsets before clustering. This was necessary because otherwise the models could cluster by language rather than by issue type.

I then sampled up to 5,000 tickets per language. This keeps the analysis computationally manageable while preserving a large enough sample for clustering and stability comparison.

### 2.3 Preprocessing

Preprocessing is implemented in [preprocessing_vectorization.py](C:/Users/Soda/Desktop/test2/preprocessing_vectorization.py).

The baseline preprocessing steps were:

- fill missing text with empty strings
- lemmatize with spaCy
- lowercase tokens
- remove stopwords
- remove punctuation
- remove digits
- remove very short tokens

This produces a normalized `processed_text` field for clustering with TF-IDF.

### 2.4 Text Representation

I created two alternative text representations:

- TF-IDF with `ngram_range=(1, 2)` and `max_features=3000`
- multilingual sentence embeddings

This design supports a direct comparison between sparse lexical features and dense semantic features, which is central to both Task 1 and the pipeline design in Task 2.a.

### 2.5 Discovering Issue Themes

Issue discovery is performed through unsupervised clustering.

- K-Means is used across multiple `k` values for comparison and stability analysis.
- HAC is used as an additional clustering baseline on embeddings.
- c-TF-IDF keywords and representative tickets are used to interpret the discovered issue categories.

This combination provides both numerical evidence and human-readable inspection.

### 2.6 Ranking Issues by Importance

The pipeline also includes a simple ranking logic for downstream use:

`Actionability Score = Volume x Mean Priority`

This ranking is implemented later in the workflow, but it is already part of the pipeline design because issue discovery is not enough on its own. A practical pipeline should also support prioritization.

## 3. Overall Judgment for My Scope

For the sections I am responsible for, the pipeline is complete at a workable level:

- Task 1 is covered through TF-IDF vs embedding comparison, K-Means analysis across `k`, HAC at `k = 15`, stability plots, and interpretability checks.
- Task 2.a is covered through EDA, preprocessing, text representation, issue discovery, and ranking design.

The strongest evidence in the current results is that embeddings clearly outperform TF-IDF for clustering customer-support issues, especially in terms of silhouette score and cluster interpretability.
