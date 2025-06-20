# Import necessary libraries for data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Load normalized count data for BM and CNS conditions
counts = pd.read_csv(
    ".../VITRO_BMvsCNS.count_normalized.txt",
    sep='\t',
    index_col=0
)[['BM', 'CNS']]  # Keep only BM and CNS columns

# Calculate log2 fold change between CNS and BM
counts['log2FC'] = np.log2((counts['CNS'] + 1) / (counts['BM'] + 1))

# Plot total counts for BM and CNS to check overall library size
plt.figure(figsize=(4,5))
sns.barplot(x=counts.columns[:2], y=counts.sum()[:2], palette=['#1f77b4', '#ff7f0e'])
plt.title("Total Normalized Counts")
plt.ylabel("Counts")
plt.xlabel("Conditions")
plt.savefig("library_sizes.png", dpi=300, bbox_inches='tight')
plt.show()

# Create MA plot: mean expression (log10) vs log2 fold change
plt.figure(figsize=(8,6))
plt.scatter(
    x=np.log10(counts[['BM','CNS']].mean(axis=1)),
    y=counts['log2FC'],
    alpha=0.3,
    s=5
)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("log10(Mean Expression)")
plt.ylabel("log2FC (CNS/BM)")
plt.title("MA Plot")
plt.savefig("MA_plot.png", dpi=300)
plt.show()

# Select top 20 most upregulated and top 20 most downregulated genes
top_genes = pd.concat([
    counts.nlargest(20, 'log2FC'),
    counts.nsmallest(20, 'log2FC')
])

# Plot heatmap for these top genes
plt.figure(figsize=(6,12))
sns.heatmap(np.log10(top_genes[['BM','CNS']]+1),
            cmap='coolwarm',
            yticklabels=True,
            xticklabels=['BM','CNS'])
plt.title("Top 40 DE Genes (log10 normalized counts)")
plt.savefig("DE_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()


# Now load full dataset with gene names included
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

file_path = '.../VITRO_BMvsCNS.count_normalized.txt'
df = pd.read_csv(file_path, sep='\t')

# Split dataset into real genes and non-targeting controls (NTC)
targeting_df = df[~df['Gene'].str.contains('Non-Targeting Control', na=False)].copy()
ntc_df = df[df['Gene'].str.contains('Non-Targeting Control', na=False)].copy()

# Calculate log2 transformed counts for both conditions
targeting_df['log2_BM'] = np.log2(targeting_df['BM'] + 1)
targeting_df['log2_CNS'] = np.log2(targeting_df['CNS'] + 1)
targeting_df['log2FC'] = targeting_df['log2_CNS'] - targeting_df['log2_BM']

# Scatter plot comparing BM and CNS expression levels
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    x='log2_BM',
    y='log2_CNS',
    data=targeting_df,
    alpha=0.5,
    s=50,
    color='steelblue'
)
plt.plot([0, 15], [0, 15], 'r--', linewidth=1.5)
plt.title('BM vs CNS: log2 Normalized Counts Comparison', fontsize=14, pad=20)
plt.xlabel('log2(BM counts + 1)')
plt.ylabel('log2(CNS counts + 1)')
plt.grid(True, alpha=0.2)

# Calculate Pearson correlation and display on plot
corr = targeting_df[['log2_BM', 'log2_CNS']].corr().iloc[0,1]
ax.text(0.05, 0.95, f'Pearson r = {corr:.2f}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('BM_vs_CNS_scatter.png', dpi=300, bbox_inches='tight')
plt.show()


# Plot distribution of log2 fold changes
plt.figure(figsize=(10, 6))
ax = sns.histplot(targeting_df['log2FC'], bins=100, kde=True, color='teal', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Distribution of log2 Fold Changes (CNS/BM)')
plt.xlabel('log2 Fold Change')
plt.ylabel('Number of sgRNAs')
plt.grid(True, alpha=0.2)

# Calculate mean and median fold changes
mean_fc = targeting_df['log2FC'].mean()
median_fc = targeting_df['log2FC'].median()
ax.text(0.05, 0.95, f'Mean = {mean_fc:.2f}\nMedian = {median_fc:.2f}',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('log2FC_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# Load gene summary data
file_path = ".../VITRO_BMvsCNS.gene_summary.txt"
df = pd.read_csv(file_path, sep="\t")

# Preview dataset columns
df.columns

# Get top 20 negatively selected genes
top_neg = df.sort_values("neg|score", ascending=True).head(20)
plt.figure(figsize=(10, 8))
sns.barplot(x="neg|score", y="id", data=top_neg, palette="Reds_r")
plt.title("Top 20 Negatively Selected Genes (Dependency Score)")
plt.xlabel("Negative Score")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()

# Get top 20 positively selected genes
top_pos = df.sort_values("pos|score", ascending=False).head(20)
plt.figure(figsize=(10, 8))
sns.barplot(x="pos|score", y="id", data=top_pos, palette="Purples")
plt.title("Top 20 Positively Selected Genes")
plt.xlabel("Positive Score")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()


# Volcano plot to visualize FDR vs Fold Change
file_path = ".../VITRO_BMvsCNS.gene_summary.txt"
df = pd.read_csv(file_path, sep="\t")

# Show basic statistics
print("\nFDR stats:\n", df["neg|fdr"].describe())
print("\nLog Fold Change stats:\n", df["neg|lfc"].describe())

# Calculate -log10(FDR)
df["-log10(FDR)"] = -np.log10(df["neg|fdr"])
df["Significant"] = (df["neg|fdr"] < 0.2) & (abs(df["neg|lfc"]) > 0.5)

plt.figure(figsize=(10, 6))
plt.scatter(df["neg|lfc"], df["-log10(FDR)"],
            c=df["Significant"].map({True: 'red', False: 'grey'}),
            alpha=0.6, s=20)
plt.axhline(-np.log10(0.2), color='blue', linestyle='--', label='FDR=0.2')
plt.axvline(-0.5, color='green', linestyle='--', label='LFC=-0.5')
plt.axvline(0.5, color='green', linestyle='--', label='LFC=0.5')
plt.title("Exploratory Volcano Plot - In Vitro")
plt.xlabel("Log2 Fold Change (neg|lfc)")
plt.ylabel("-log10(FDR)")
plt.legend()
plt.tight_layout()
plt.show()


# Select top 20 genes with highest absolute fold change
top_genes = df.reindex(df["neg|lfc"].abs().sort_values(ascending=False).index).head(20)
print(top_genes[["id", "neg|lfc", "neg|fdr"]])

# Bar plot of top 20 genes based on absolute log2 fold change
plt.figure(figsize=(10, 6))
sns.barplot(x="neg|lfc", y="id", data=top_genes, palette="vlag")
plt.title("Top 20 Genes by Absolute Log Fold Change (In Vitro)")
plt.xlabel("Log2 Fold Change")
plt.ylabel("Gene ID")
plt.tight_layout()
plt.show()


# Simple gene prioritization scatter plot again
plt.figure(figsize=(10, 6))
plt.scatter(df["neg|lfc"], df["-log10(FDR)"], alpha=0.5)
plt.title("Gene Prioritization Plot (in vitro BM vs CNS)")
plt.xlabel("Log Fold Change (neg|lfc)")
plt.ylabel("-log10(FDR)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Heatmap of dependency scores for selected genes from DepMap
file_path = ".../CRISPR_gene_effect.csv"
depmap_df = pd.read_csv(file_path)
depmap_df = depmap_df.set_index("DepMap_ID")

# Select genes of interest
top_genes = ["STRN4", "LRIG2", "GATAD2A", "MMP11", "ERP27"]
matched_cols = [col for col in depmap_df.columns if any(gene in col for gene in top_genes)]

# Extract data and prepare heatmap
heatmap_df = depmap_df[matched_cols].T
heatmap_df.index = [col.split(" ")[0] for col in heatmap_df.index]

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_df, cmap="vlag", center=0)
plt.title("DepMap CRISPR Gene Dependency Scores")
plt.xlabel("Cell Lines")
plt.ylabel("Genes")
plt.tight_layout()
plt.show()



