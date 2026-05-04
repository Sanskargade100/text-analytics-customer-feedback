# EMATM0067: Task 4 - Mining Insights from Customer Feedback

**Team:** 9  
**Members:** Jiahong Chen;Xiaoran Yang;Tanaphan Kummaraphat;Abrish Aaditya Sivakumar Brindha;Sanskar Gade
**Course:** MSc Text Analytics (EMATM0067)  
**Task:** Task 4 - Mining Customer Feedback  

## Project Overview

This repository contains our final coursework submission for Task 4. The aim of this project is to process unstructured customer-support tickets in English and German, extracting meaningful, understandable, and actionable issue categories that can inform service improvements.

Rather than just grouping text, our pipeline transforms raw ticket data into a ranked list of customer pain points by evaluating two analytic axes:
1. **Text Representation (Axis 1):** Comparing sparse lexical features (TF-IDF) against dense semantic representations (multilingual sentence embeddings).
2. **Topic Modelling (Axis 2):** Comparing traditional bag-of-words topic discovery (LDA) against an embedding-based approach (BERTopic).

## Our Pipeline

The project follows a modular, 5-step workflow:
1. **Data Exploration & Sampling:** Merging subject and body fields, cleaning out-of-domain queues, and balancing the dataset to 5,000 tickets per language.
2. **Linguistic Preprocessing:** Using spaCy to lowercase, remove punctuation, digits, and stopwords, followed by lemmatisation. 
3. **Axis 1 (Representation):** Clustering with K-Means (and HAC) to demonstrate that semantic embeddings capture diverse support issues better than TF-IDF.
4. **Axis 2 (Theme Discovery):** Applying LDA and BERTopic. While LDA scored better on automated coherence metrics, BERTopic generated topics that were much easier to interpret and map to real-world support queues.
5. **Issue Ranking:** Calculating an 'Actionability Score' by combining topic volume with ticket priority, ensuring that the highest-ranked issues reflect both frequency and operational urgency.

## Repository Structure

- **`Final_Code.ipynb`**: The main Jupyter Notebook containing the full end-to-end pipeline, from data loading to final issue ranking.
- **`EMATM0067_Task04_Team09_Report.pdf`**: Our final written report detailing our methodology, experiments, and conclusions.
- **`results/`**: Directory containing all generated outputs, including stability plots, topic coherence scores, and the final ranked issue CSVs.
- **`Draft/` & `Overleaf/`**: Folders containing our working drafts, initial individual contributions, and LaTeX source files.

## How to Run

1. Ensure you have Python 3 installed alongside the required libraries: `pandas`, `scikit-learn`, `spacy`, `bertopic`, `sentence-transformers`, `gensim`, `datasets`, `matplotlib`, and `seaborn`. 
2. Download the required spaCy models:
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download de_core_news_sm
   ```
3. Open `Final_Code.ipynb` and run the cells sequentially. The notebook will automatically download the dataset from Hugging Face and generate all outputs in the `results/` folder.

## Key Findings

- **Embeddings outperform TF-IDF:** Sentence embeddings are far more robust at clustering short, noisy customer tickets where terminology varies.
- **Interpretability over Coherence:** Although LDA achieved higher automated coherence scores, BERTopic produced much more actionable and nameable clusters.
- **Actionable Ranking:** Sorting issues solely by volume is insufficient; factoring in ticket priority highlighted critical but smaller topics (like Medical Data Security and SaaS platform performance) that require immediate attention.
