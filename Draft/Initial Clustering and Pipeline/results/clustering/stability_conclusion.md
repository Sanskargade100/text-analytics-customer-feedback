# Clustering Stability and k-Selection

This note summarizes the clustering analysis used for Task 4 Section 1.

## Methods

The analysis compares:

- K-Means on TF-IDF vectors
- K-Means on sentence embeddings
- HAC on embeddings at `k = 15`

K-Means was tested for odd values of `k` from 5 to 25 on both English and German subsets.

## Stability Across k

The metric files show that embeddings are consistently stronger than TF-IDF.

- English TF-IDF silhouette scores stay very low, roughly between `0.00` and `0.04`.
- German TF-IDF is weaker still, with several negative silhouette scores.
- English embeddings stay around `0.20` to `0.24`, which indicates much better cluster separation.
- German embeddings stay around `0.11` to `0.14`, again clearly better than TF-IDF.

Across the tested range, the embedding curves are more stable and more usable for interpretation than TF-IDF.

## k Selection

`k = 15` was selected as the working value for interpretation.

This is not the absolute best silhouette point in every case, but it is a reasonable compromise because it:

- preserves acceptable stability on embeddings
- gives more detailed issue categories than very small `k`
- remains interpretable when inspecting cluster keywords and representative tickets

## Coherence and Interpretability

Inspection of the cluster reports shows several coherent issue groups, especially in the embedding-based English and German outputs.

Examples include:

- medical data security breaches
- billing and payment errors
- network or VPN connectivity issues
- software crashes and outages
- SaaS or integration support requests

Some clusters remain noisy, especially under TF-IDF-style lexical grouping and in a few German clusters with encoding or mixed-language artifacts. Even so, the embedding-based clusters are sufficiently coherent for issue discovery.

## Conclusion

For this dataset, embeddings are the better representation for clustering customer-support issues. K-Means on embeddings with `k = 15` provides the best practical balance between stability, granularity, and interpretability, while HAC serves as a useful comparison baseline.
