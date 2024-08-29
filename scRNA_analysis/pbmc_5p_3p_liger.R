# DATA INEGRATION: Liger

library(Seurat)
library(tidyverse)
library(Matrix)
library(ggplot2)
library(rliger)
library(SingleR)
library(celldex)
library(RColorBrewer)
library(SeuratWrappers)
library(SingleCellExperiment)
library(HGNChelper)
library(RcppPlanc)

data_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\dataset\\"
plots_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\plots\\"
rds_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\saved_rds\\"
sctype_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\sctype_files\\" 

# Load RDS file
seurat_obj = readRDS(file = paste0(rds_dir, "seurat_obj_loaded.rds"))

# QC
seurat_obj[["percent.mt"]] = PercentageFeatureSet(seurat_obj, pattern = "^MT-")
seurat_obj = subset(seurat_obj, subset = nFeature_RNA>300 & nFeature_RNA<2500 & percent.mt<5)

# Convert the Seurat object to LIGER object
liger_obj = seuratToLiger(seurat_obj)

# Normalize, scale, and select variable genes
liger_obj = liger_obj %>%
  rliger::normalize() %>%
  rliger::selectGenes() %>%
  rliger::scaleNotCenter()

# Perform integrative non-negative matrix factorization
liger_obj = runIntegration(liger_obj, k = 20)

# Quantile normalization
liger_obj = quantileNorm(liger_obj)

# Perform Louvain clustering
liger_obj = louvainCluster(liger_obj, resolution = 0.8)

# UMAP visualization
liger_obj = runUMAP(liger_obj, nNeighbors = 30, min_dist = 0.3)
plotByDatasetAndCluster(liger_obj)

# Save the integrated Liger object
saveRDS(liger_obj, file = paste0(rds_dir, "liger_obj_integrated.rds"))
liger_obj = readRDS(file=paste0(rds_dir, "liger_obj_integrated.rds"))

# Convert Liger object to Seurat object
seurat_obj = ligerToSeurat(liger_obj)

# Count table
count_table = table(seurat_obj@meta.data$louvain_cluster, seurat_obj@meta.data$SampleID)
print(count_table)

# Define custom function to plot integrated clusters
plot_integrated_clusters <- function(seurat_obj) {
  ggplot(seurat_obj@meta.data, aes(x = louvain_cluster, fill = SampleID)) +
    geom_bar(position = "stack") +
    labs(title = "Cluster Distribution Among Samples", x = "Cluster", y = "Number of Cells") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_discrete(name = "Sample ID")
}
plot_integrated_clusters(seurat_obj)

# Differentially expressed genes in the integrated dataset

# To find clusters we will perform NormalizeData, ScaleData, PCA using Seurat packages
seurat_obj = NormalizeData(seurat_obj)
seurat_obj = FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
seurat_obj = ScaleData(seurat_obj)
seurat_obj = RunPCA(seurat_obj, npcs = 30, verbose = F)
seurat_obj = FindNeighbors(seurat_obj, dims = 1:30)
seurat_obj = FindClusters(seurat_obj, resolution = 0.8)

# Find markers for all clusters

seurat_obj = JoinLayers(seurat_obj)

seurat_obj.markers = FindAllMarkers(seurat_obj, only.pos = TRUE)
seurat_obj.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1)

top_genes_per_cluster = seurat_obj.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1) %>%
  slice_head(n=5) %>%
  ungroup() -> top5

# Print top 5 genes for each cluster
top_genes_per_cluster %>%
  select(cluster, gene, avg_log2FC) %>%
  group_by(cluster) %>%
  summarise(top_genes = paste(gene, collapse = ", ")) %>%
  print()

DoHeatmap(seurat_obj, features=top5$gene) + NoLegend()

# Assign cell type to clusters

# (1) SingleR annotation

# Get reference datasets from celldex package
monaco.ref = celldex::MonacoImmuneData()

# Convert Seurat object to SingleCellExperiment (SCE) for convenience
sce = as.SingleCellExperiment(DietSeurat(seurat_obj))

# Run SingleR for cell type annotation
monaco.main = SingleR(test = sce, assay.type.test = 1, ref = monaco.ref, labels = monaco.ref$label.main)
monaco.fine = SingleR(test = sce, assay.type.test = 1, ref = monaco.ref, labels = monaco.ref$label.fine)

# Summary of general cell type annotations
table(monaco.main$pruned.labels)

# Summary of finer cell type annotations
table(monaco.fine$pruned.labels)

# Add annotations to the Seurat object metadata
seurat_obj@meta.data$monaco.main = monaco.main$pruned.labels
seurat_obj@meta.data$monaco.fine = monaco.fine$pruned.labels

# Visualize the main annotations
seurat_obj = SetIdent(seurat_obj, value = "monaco.main")
DimPlot(seurat_obj, label = TRUE, repel = TRUE, label.size = 4)

# (2) SCTYPE annotation

source(paste0(sctype_dir, "gene_sets_prepare.R"))
source(paste0(sctype_dir, "sctype_score_.R"))

# Built in cell-type-specific gene set database
gs_list = gene_sets_prepare(paste0(sctype_dir, "ScTypeDB_short.xlsx"), "Immune system")

# Load scaled data, when using Seurat, use "RNA" slot with 'scale.data' by default
scRNAseqData = GetAssayData(object = seurat_obj[["RNA"]], slot = "scale.data")

# Assign cell types
es.max = sctype_score(scRNAseqData = scRNAseqData, scaled = TRUE, gs = gs_list$gs_positive, gs2 = gs_list$gs_negative)

# Merge by cluster
cL_resutls = do.call("rbind", lapply(unique(seurat_obj@meta.data$seurat_clusters), function(cl)
{
  es.max.cl = sort(rowSums(es.max[ ,rownames(seurat_obj@meta.data[seurat_obj@meta.data$seurat_clusters==cl, ])]), decreasing = !0)
  head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl, ncells = sum(seurat_obj@meta.data$seurat_clusters==cl)), 10)
}
))
sctype_scores = cL_resutls %>% 
  group_by(cluster) %>% 
  top_n(n = 1, wt = scores)  

# Set low-confident (low ScType score) clusters to "unknown"
sctype_scores$type[as.numeric(as.character(sctype_scores$scores)) < sctype_scores$ncells/4] = "Unknown"
print(sctype_scores[,1:3])

# Add annotations to the Seurat object metadata
seurat_obj@meta.data$sctype_classification = ""
for(j in unique(sctype_scores$cluster))
{
  cl_type = sctype_scores[sctype_scores$cluster==j,]
  seurat_obj@meta.data$sctype_classification[seurat_obj@meta.data$seurat_clusters == j] = as.character(cl_type$type[1])
}

# Visualise the annotations
seurat_obj = SetIdent(seurat_obj, value = "sctype_classification")
DimPlot(seurat_obj, label = TRUE, repel = TRUE, label.size = 4)

# Save as RDS file
saveRDS(seurat_obj, file=paste0(rds_dir, "seurat_obj_liger_annotated.rds"))
