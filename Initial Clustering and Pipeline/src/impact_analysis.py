import pandas as pd
from update_labels import apply_labels, en_labels, de_labels

priority_map = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Critical': 4
}

def issue_label(row):
    business_label = row.get('business_label')
    if pd.notna(business_label) and str(business_label).strip():
        return str(business_label)
    return str(row['top_words'])[:60] + "..."

def write_summary():
    en_ranked = pd.read_csv('results/analysis/ranked_issues_en.csv')
    de_ranked = pd.read_csv('results/analysis/ranked_issues_de.csv')

    with open('results/analysis/summary_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("Task 4: Mining Insights Summary\n")
        f.write("===============================\n\n")
        f.write("Top 5 Issues in English (by Actionability Score):\n")
        for _, row in en_ranked.head(5).iterrows():
            f.write(
                f"- Cluster {row['cluster_id']}: {issue_label(row)} "
                f"(Score: {row['actionability_score']:.1f})\n"
            )

        f.write("\nTop 5 Issues in German (by Actionability Score):\n")
        for _, row in de_ranked.head(5).iterrows():
            f.write(
                f"- Cluster {row['cluster_id']}: {issue_label(row)} "
                f"(Score: {row['actionability_score']:.1f})\n"
            )

def analyze_impact(lang):
    print(f"Analyzing impact for {lang}...")
    df = pd.read_csv(f'results/interpretation/df_{lang}_labeled.csv')
    report = pd.read_csv(f'results/interpretation/cluster_report_{lang}.csv')
    
    # Map priority
    df['priority_val'] = df['priority'].map(priority_map).fillna(1)
    
    # Calculate impact metrics per cluster
    stats = df.groupby('cluster').agg({
        'priority_val': 'mean',
        'subject': 'count' 
    }).rename(columns={'priority_val': 'avg_priority', 'subject': 'volume'})
    
    # Merge with cluster info
    final_report = report.merge(stats, left_on='cluster_id', right_index=True)
    
    # Calculate "Actionability Score" (Volume * Avg Priority)
    final_report['actionability_score'] = final_report['volume'] * final_report['avg_priority']
    
    # Rank by score
    final_report = final_report.sort_values(by='actionability_score', ascending=False)
    
    # Save final ranked report
    final_report.to_csv(f'results/analysis/ranked_issues_{lang}.csv', index=False)
    
    return final_report

print("Running impact analysis...")
analyze_impact('en')
analyze_impact('de')
apply_labels('en', en_labels)
apply_labels('de', de_labels)
write_summary()

print("Analysis and ranking complete.")
