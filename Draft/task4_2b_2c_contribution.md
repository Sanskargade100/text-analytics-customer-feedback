# Task4 2.b and 2.c Contribution

This file contains my contribution to **Task 4.2.b** and **Task 4.2.c** of the group coursework.

My part focuses on:
- selecting two important analytic axes for the customer-support ticket pipeline
- defining the compared approaches under each axis
- motivating each comparison with clear hypotheses
- connecting the later pipeline design to the initial clustering analysis already completed in Task 4.1 and Task 4.2.a

## What I did

I completed the following steps:

1. Chose the two main analytic axes
- reviewed the coursework brief for Task 4
- checked the existing pipeline outputs from Task 4.1 and Task 4.2.a
- selected two analytic axes that extend the current pipeline without repeating the earlier clustering work

2. Defined Axis 1: text representation
- used the existing comparison between TF-IDF and multilingual sentence embeddings
- treated this axis as the main comparison for how ticket text is represented before clustering
- linked the choice to the observed stability plots and silhouette-score differences across English and German

3. Defined Axis 2: issue labelling and summarisation
- fixed the clustering setting to the already selected main configuration: embeddings + K-Means + k = 15
- treated the second axis as a downstream comparison of how discovered issue clusters are interpreted
- avoided reusing K-Means versus HAC as the main second axis, since that comparison had already been covered in the initial clustering stage

4. Compared alternative approaches under each axis

### Axis 1: text representation
- TF-IDF vectors
- multilingual sentence embeddings

### Axis 2: issue labelling and summarisation
- keywords only
- keywords plus representative tickets
- business labels / short summarised issue descriptions

5. Wrote hypotheses for each comparison
- embeddings should outperform TF-IDF for issue clustering because customer-support tickets are short, noisy, and lexically diverse
- TF-IDF should remain more transparent at the lexical level, but should produce weaker issue separation
- richer labelling strategies should improve interpretability and actionability, even when the underlying clusters remain unchanged

## Main decisions from this part

The final design for my scope is:

- **Axis 1:** text representation (TF-IDF vs embeddings)
- **Axis 2:** issue labelling / summarisation method

For the second axis, the cluster structure is kept fixed and only the interpretation layer is compared. This makes the comparison more useful for the final task objective, which is to identify and summarise customer issues into actionable insights.

## Suggested report text for Task 4.2.b

### 2.b Chosen analytic axes

To investigate how design choices affect the quality of insights extracted from customer support tickets, we focus on two analytic axes. The first axis is **text representation**, comparing TF-IDF and multilingual sentence embeddings. This axis examines how different representations affect the coherence and usefulness of the discovered issue categories. The second axis is **issue labelling and summarisation**, which compares different ways of turning discovered clusters into human-readable issue categories. We choose this second axis instead of reusing K-Means versus HAC, because clustering methods were already examined in the initial analysis stage, where K-Means was tested across multiple values of \(k\), HAC was used as a supplementary comparison, and embeddings with K-Means at \(k=15\) were retained as the main working configuration.

These two axes were selected because they affect two different parts of the pipeline. Text representation influences how tickets are grouped into issue categories, while labelling and summarisation influence how useful those categories are for interpretation and actionable reporting.

## Suggested report text for Task 4.2.c

### 2.c Compared approaches and hypotheses

For **Axis 1**, we compare **TF-IDF vectors** and **multilingual sentence embeddings**. Our hypothesis is that embeddings will produce more coherent and stable issue clusters, because customer-support tickets often describe similar problems using varied wording, abbreviations, and paraphrases. In contrast, TF-IDF is expected to remain more transparent in lexical terms, but weaker for semantic grouping.

For **Axis 2**, we compare three labelling strategies applied to the fixed embedding-based K-Means clusters at \(k=15\). The first is **keywords only**, where each cluster is described using its top c-TF-IDF terms. The second is **keywords plus representative tickets**, where the keywords are supplemented by a few tickets closest to the cluster centroid. The third is **business labels / short summarised issue descriptions**, which convert cluster evidence into concise issue names that are easier to communicate in a product or service context.

Our hypothesis for Axis 2 is that **keywords-only labels will be the fastest but least reliable**, especially for noisy clusters; **keywords plus representative tickets will improve interpretability** by giving concrete context; and **business-oriented summaries will be the most actionable**, although they involve more post-hoc human interpretation.

## Files and outputs used for this part

The most relevant existing outputs for this contribution are:
- `results/clustering/metrics_en_tfidf.csv`
- `results/clustering/metrics_en_emb.csv`
- `results/clustering/metrics_de_tfidf.csv`
- `results/clustering/metrics_de_emb.csv`
- `results/plots/stability_en_tfidf.png`
- `results/plots/stability_en_emb.png`
- `results/plots/stability_de_tfidf.png`
- `results/plots/stability_de_emb.png`
- `results/interpretation/cluster_report_en.csv`
- `results/interpretation/cluster_report_de.csv`
- `results/analysis/ranked_issues_en.csv`
- `results/analysis/ranked_issues_de.csv`

## Notes for the next person

If you continue from this part, the most useful next steps are:
- use the stability plots and metric files to support the Axis 1 comparison in the report
- use the cluster reports to build a small comparison table for Axis 2, showing how the same cluster looks under different labelling strategies
- connect the Axis 2 comparison to the later evaluation section by discussing coherence, interpretability, and actionability

## Limitations

This contribution mainly defines and motivates the experimental axes. The first axis already has strong quantitative evidence from the earlier clustering analysis, but the second axis depends more on qualitative comparison of cluster descriptions. This means the issue-labelling comparison is lighter-weight and more interpretive than the representation comparison.
