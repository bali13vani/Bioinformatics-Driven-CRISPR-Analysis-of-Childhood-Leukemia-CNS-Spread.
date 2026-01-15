# Install packages (run once if needed)
# !pip install gseapy pandas

import pandas as pd
import gseapy as gp
import os

# ================================
# 1. Load expression data
# ================================
data = pd.read_csv("VITRO_BMvsCNS.gene_summary.txt", sep="\t")

# Keep only genes with logFC values
data = data.dropna(subset=["neg|lfc"])

# Create ranked gene list
ranked_data = data[["id", "neg|lfc"]]
ranked_data.columns = ["Gene", "Score"]
ranked_data = ranked_data.sort_values(by="Score", ascending=False)

ranking_file = "vitro_ranked_genes.rnk"
ranked_data.to_csv(ranking_file, sep="\t", index=False, header=False)

# ================================
# 2. Run GSEA (Hallmark pathways)
# ================================
gene_sets = "h.all.v2025.1.Hs.symbols.gmt"

outdir = "vitro_gsea_results"
os.makedirs(outdir, exist_ok=True)

gsea_results = gp.prerank(
    rnk=ranking_file,
    gene_sets=gene_sets,
    outdir=outdir,
    permutation_num=1000,
    seed=42,
    format="png"
)

# ================================
# 3. Extract top pathways
# ================================
results_table = gsea_results.res2d
top_pathways = results_table.sort_values("FDR q-val").head(10)

top_pathways
