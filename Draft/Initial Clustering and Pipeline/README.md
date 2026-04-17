# Task4 1 and 2.a Contribution

This folder contains my work for Task 4.1 and Task 4.2.a of the group coursework.

My part focuses on:
- clustering customer support tickets to identify issue types
- checking cluster coherence, interpretability, and stability across different k values
- building the early part of the text analytics pipeline, including data exploration, preprocessing, text representation, issue discovery, and issue ranking

## What I did

I completed the following steps:

1. Data exploration
- loaded the customer support ticket dataset
- combined subject and body into one text field
- split the dataset into English and German subsets
- sampled 5,000 tickets for each language
- checked missing values, duplicates, priority distribution, type distribution, and text length statistics

2. Preprocessing and text representation
- applied baseline preprocessing with SpaCy lemmatization
- removed stopwords, punctuation, and digits
- generated both TF-IDF vectors and dense embeddings

3. Clustering analysis
- compared TF-IDF and embedding representations
- ran K-Means across multiple k values
- used SSE and silhouette score to examine stability
- ran HAC as a supplementary comparison on embeddings

4. Cluster interpretation
- extracted cluster keywords using c-TF-IDF
- used language-specific stopwords for English and German
- selected representative tickets from within each cluster
- created interpretable cluster reports

5. Ranking and labeling
- ranked issue clusters using volume and mean priority
- added manual business labels after reviewing keywords and representative tickets

## Main files

### Scripts
- `data_exploration.py`  
  Performs EDA, language split, sampling, and summary statistics.

- `preprocessing_vectorization.py`  
  Applies preprocessing and creates TF-IDF and embedding representations.

- `clustering_analysis.py`  
  Runs clustering experiments and saves evaluation metrics.

- `interpret_clusters.py`  
  Extracts cluster keywords and representative examples.

- `visualize_metrics.py`  
  Creates stability plots across different k values.

- `impact_analysis.py`  
  Ranks issue types based on cluster size and average priority.

- `update_labels.py`  
  Adds manual business labels to the ranked cluster outputs.

### Report
- `pipeline_report.md`  
  Short explanation of the pipeline design, design choices, and final decisions for Task 4.2.a.

## Results folder

The `results/` folder contains the main outputs from my part of the project.

### Main outputs
- `results/data_stats.txt`  
  Summary of EDA findings.

- `results/clustering/metrics_*.csv`  
  Clustering evaluation results across different k values.

- `results/plots/`  
  Stability plots for TF-IDF and embeddings in English and German.

- `results/interpretation/cluster_report_en.csv`  
  Cluster interpretation report for English tickets.

- `results/interpretation/cluster_report_de.csv`  
  Cluster interpretation report for German tickets.

- `results/interpretation/df_en_labeled.csv`  
  English tickets with assigned cluster labels.

- `results/interpretation/df_de_labeled.csv`  
  German tickets with assigned cluster labels.

- `results/analysis/ranked_issues_en.csv`  
  Ranked English issue clusters.

- `results/analysis/ranked_issues_de.csv`  
  Ranked German issue clusters.

- `results/analysis/summary_comparison.txt`  
  Short comparison summary of the final discovered issue types.

## Final decisions from this part

Based on the comparison results:
- embeddings were more useful than TF-IDF for issue clustering
- K-Means was used as the main clustering method
- k = 15 was selected as the final setting
- HAC was kept as a supplementary comparison, not the main method

## Notes for the next person

If you continue from this part, the most useful files to start with are:
- `pipeline_report.md`
- `results/clustering/metrics_en_emb.csv`
- `results/clustering/metrics_de_emb.csv`
- `results/interpretation/cluster_report_en.csv`
- `results/interpretation/cluster_report_de.csv`
- `results/analysis/ranked_issues_en.csv`
- `results/analysis/ranked_issues_de.csv`

These files show:
- how the clusters were selected
- what each cluster means
- which issue types seem most important

## Limitations

This version uses baseline preprocessing. It removes digits and punctuation, which helps reduce noise, but it may also remove useful technical details such as error codes, version numbers, or order IDs.

Also, the final business labels were assigned manually after cluster inspection, so they should be understood as post-hoc interpretive labels rather than fully automatic outputs.