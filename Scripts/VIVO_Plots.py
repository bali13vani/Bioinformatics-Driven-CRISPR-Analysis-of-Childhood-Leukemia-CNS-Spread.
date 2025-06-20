# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Load normalized counts data for BM and CNS from the file
counts = pd.read_csv(
    ".../VIVO_BMvsCNS.count_normalized.txt",
    sep='\t', 
    index_col=0
)[['BM', 'CNS']]  # Select only the BM and CNS columns

# Calculate log2 fold change between CNS and BM (add 1 to avoid division by zero)
counts['log2FC'] = np.log2((counts['CNS'] + 1) / (counts['BM'] + 1))

# Create a barplot to compare total counts in BM and CNS
plt.figure(figsize=(4,5))
sns.barplot(
    x=counts.columns[:2],  # BM and CNS
    y=counts.sum()[:2],     # Sum total counts for both conditions
    palette=['#1f77b4', '#ff7f0e']  # Colors for bars
)
plt.title("Total Normalized Counts")
plt.ylabel("Counts")
plt.xlabel("Conditions")
plt.savefig("library_sizes.png", dpi=300, bbox_inches='tight')
plt.show()

# Create an MA plot: log10 mean expression vs log2 fold change
plt.figure(figsize=(8,6))
plt.scatter(
    x=np.log10(counts[['BM','CNS']].mean(axis=1)),
    y=counts['log2FC'],
    alpha=0.3,
    s=5
)
plt.axhline(0, color='red', linestyle='--')  # Add horizontal line at log2FC=0
plt.xlabel("log10(Mean Expression)")
plt.ylabel("log2FC (CNS/BM)")
plt.title("MA Plot")
plt.savefig("MA_plot.png", dpi=300)
plt.show()

# Select top 20 upregulated and top 20 downregulated genes based on log2FC
top_genes = pd.concat([
    counts.nlargest(20, 'log2FC'),
    counts.nsmallest(20, 'log2FC')
])

# Plot heatmap for these top 40 genes
plt.figure(figsize=(6,12))
sns.heatmap(
    np.log10(top_genes[['BM','CNS']]+1), 
    cmap='coolwarm', 
    yticklabels=True, 
    xticklabels=['BM','CNS']
)
plt.title("Top 40 DE Genes (log10 normalized counts)")
plt.savefig("DE_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()


# Now load full dataset (with gene names included) for further analysis
file_path = '.../VIVO_BMvsCNS.count_normalized.txt'
df = pd.read_csv(file_path, sep='\t')

# Split into targeting sgRNAs (real genes) and non-targeting controls (NTCs)
targeting_df = df[~df['Gene'].str.contains('Non-Targeting Control', na=False)].copy()
ntc_df = df[df['Gene'].str.contains('Non-Targeting Control', na=False)].copy()

# Log2 transform counts for both conditions
targeting_df['log2_BM'] = np.log2(targeting_df['BM'] + 1)
targeting_df['log2_CNS'] = np.log2(targeting_df['CNS'] + 1)
targeting_df['log2FC'] = targeting_df['log2_CNS'] - targeting_df['log2_BM']

# Scatter plot to compare BM and CNS counts in log2 scale
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    x='log2_BM', 
    y='log2_CNS', 
    data=targeting_df, 
    alpha=0.5, 
    s=50,
    color='steelblue'
)
plt.plot([0, 15], [0, 15], 'r--', linewidth=1.5)  # diagonal reference line
plt.title('BM vs CNS: log2 Normalized Counts Comparison', fontsize=14, pad=20)
plt.xlabel('log2(BM counts + 1)', fontsize=12, labelpad=10)
plt.ylabel('log2(CNS counts + 1)', fontsize=12, labelpad=10)
plt.grid(True, alpha=0.2)

# Calculate Pearson correlation between BM and CNS
corr = targeting_df[['log2_BM', 'log2_CNS']].corr().iloc[0,1]
ax.text(0.05, 0.95, f'Pearson r = {corr:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('BM_vs_CNS_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot distribution of log2 fold changes (CNS vs BM)
plt.figure(figsize=(10, 6))
ax = sns.histplot(
    targeting_df['log2FC'], 
    bins=100, 
    kde=True, 
    color='teal',
    alpha=0.7
)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Distribution of log2 Fold Changes (CNS/BM)', fontsize=14, pad=20)
plt.xlabel('log2 Fold Change', fontsize=12, labelpad=10)
plt.ylabel('Number of sgRNAs', fontsize=12, labelpad=10)
plt.grid(True, alpha=0.2)

# Display mean and median log2 fold change on the plot
mean_fc = targeting_df['log2FC'].mean()
median_fc = targeting_df['log2FC'].median()
ax.text(0.05, 0.95, f'Mean = {mean_fc:.2f}\nMedian = {median_fc:.2f}', 
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('log2FC_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# Load gene summary file for gene-level analysis
df = pd.read_csv(".../VIVO_BMvsCNS.gene_summary.txt", sep="\t")

# Remove any rows without FDR values
df = df.dropna(subset=["neg|fdr"])
df["-log10(FDR)"] = -np.log10(df["neg|fdr"])

# Sort by significance and plot top 20 essential genes
df_sorted = df.sort_values("-log10(FDR)", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="-log10(FDR)", y="id", data=df_sorted.head(20), palette="viridis")
plt.title("Top 20 Essential Genes (in vivo BM vs CNS)")
plt.xlabel("-log10(FDR)")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()

# Select top 20 negatively selected genes based on negative score
top_neg = df.sort_values("neg|score", ascending=True).head(20)
plt.figure(figsize=(10, 8))
sns.barplot(x="neg|score", y="id", data=top_neg, palette="Reds_r")
plt.title("Top 20 Negatively Selected Genes (Dependency Score)")
plt.xlabel("Negative Score")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()

# Select top 20 positively selected genes based on positive score
top_pos = df.sort_values("pos|score", ascending=False).head(20)
plt.figure(figsize=(10, 8))
sns.barplot(x="pos|score", y="id", data=top_pos, palette="Purples")
plt.title("Top 20 Positively Selected Genes")
plt.xlabel("Positive Score")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()


# Create exploratory volcano plot to visualize fold change vs FDR
df = df.dropna(subset=["neg|fdr", "neg|lfc"])
df["-log10(FDR)"] = -np.log10(df["neg|fdr"])
df["Significant"] = (df["neg|fdr"] < 0.2) & (abs(df["neg|lfc"]) > 0.5)

plt.figure(figsize=(10, 6))
plt.scatter(
    df["neg|lfc"], df["-log10(FDR)"], 
    c=df["Significant"].map({True: 'red', False: 'grey'}),
    alpha=0.6, s=20
)
plt.axhline(-np.log10(0.2), color='blue', linestyle='--', label='FDR=0.2')
plt.axvline(-0.5, color='green', linestyle='--', label='LFC=-0.5')
plt.axvline(0.5, color='green', linestyle='--', label='LFC=0.5')
plt.title("Exploratory Volcano Plot - In Vivo")
plt.xlabel("Log2 Fold Change (neg|lfc)")
plt.ylabel("-log10(FDR)")
plt.legend()
plt.tight_layout()
plt.show()


# Select top 20 genes based on absolute log fold change
top_genes = df.reindex(df["neg|lfc"].abs().sort_values(ascending=False).index).head(20)
plt.figure(figsize=(10, 6))
sns.barplot(x="neg|lfc", y="id", data=top_genes, palette="vlag")
plt.title("Top 20 Genes by Absolute Log Fold Change (In Vivo)")
plt.xlabel("Log2 Fold Change")
plt.ylabel("Gene ID")
plt.tight_layout()
plt.show()


# Simple gene prioritization scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='neg|lfc', y='-log10(FDR)', alpha=0.7)
plt.title("Gene Prioritization Plot (in vivo BM vs CNS)")
plt.xlabel("Log Fold Change (neg|lfc)")
plt.ylabel("-log10(FDR)")
plt.grid(True)
plt.tight_layout()
plt.show()



