import pandas as pd

en_labels = {
    0: "Project Management Tool Integration",
    1: "Digital Marketing Engagement Decline",
    2: "General/Miscellaneous (Noisy)",
    3: "Network Connectivity & Device Sync",
    4: "Medical Data Security Breaches",
    5: "SaaS Platform Integration Docs",
    6: "Hospital Data Protection",
    7: "Digital Strategy & Brand Growth",
    8: "Marketing Campaign Performance",
    9: "Billing & Subscription Errors",
    10: "Investment Analytics Optimization",
    11: "System Crashes & Compatibility",
    12: "SaaS Platform Performance",
    13: "Financial Data Discrepancies",
    14: "Server Outages & Peak Load"
}

de_labels = {
    0: "Software Bug Reports",
    1: "General/Miscellaneous (Noisy)",
    2: "VPN Connectivity Issues",
    3: "System Crashes (Data Platform)",
    4: "Digital Marketing Strategy",
    5: "Critical System Outages (Urgent)",
    6: "Billing & Payment Errors",
    7: "Medical Data Integrity",
    8: "Marketing Tool Performance",
    9: "SaaS Integration Support",
    10: "General Inquiry/Access Support",
    11: "Cybersecurity & Firewall",
    12: "Maintenance & Update Delays",
    13: "Investment Analytics Improvement",
    14: "Data Analysis Tool Bugs"
}

def apply_labels(lang, label_map):
    path = f'results/analysis/ranked_issues_{lang}.csv'
    df = pd.read_csv(path)
    df['business_label'] = df['cluster_id'].map(label_map)
    # Reorder columns to put label at front
    cols = ['cluster_id', 'business_label'] + [c for c in df.columns if c not in ['cluster_id', 'business_label']]
    df = df[cols]
    df.to_csv(path, index=False)
    print(f"Applied business labels to {lang} ranked issues.")

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

if __name__ == "__main__":
    apply_labels('en', en_labels)
    apply_labels('de', de_labels)
    write_summary()
