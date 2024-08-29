# DATA INEGRATION: CCA

library(Seurat)
library(tidyverse)
library(Matrix)
library(ggplot2)
library(harmony)
library(rliger)
library(SingleR)
library(celldex)
library(RColorBrewer)
library(SeuratWrappers)
library(SingleCellExperiment)
library(HGNChelper)

data_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\dataset\\"
plots_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\plots\\"
rds_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\saved_rds\\"
sctype_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\sctype_files\\" 

# Load RDS file
seurat_obj = readRDS(file = paste0(rds_dir, "seurat_obj_loaded.rds"))

set.seed(22222)

# Split the Seurat object by Sample ID
split_seurat = SplitObject(seurat_obj, split.by = "SampleID")

# Normalize and identify variable features for each dataset independently
split_seurat = lapply(X = split_seurat, FUN = function(x)
{
  x[["percent.mt"]] = PercentageFeatureSet(x, pattern = "^MT-")
  x = subset(x, subset = nFeature_RNA>300 & nFeature_RNA<2500 & percent.mt<5)
  x = NormalizeData(x)
  x = FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
  return(x)
})

# Select the most variable features to use for integration
integ_features = SelectIntegrationFeatures(object.list = split_seurat, nfeatures = 2000)

# Find integration anchors
integ_anchors = FindIntegrationAnchors(object.list = split_seurat, anchor.features = integ_features)

# Integrate data
seurat_integrated = IntegrateData(anchorset = integ_anchors)

# Run the standard workflow for visualization and clustering
seurat_integrated = ScaleData(seurat_integrated)
seurat_integrated = RunPCA(seurat_integrated, npcs = 30)

# Find how many PCs should be included for downstream analysis
pca = seurat_integrated[["pca"]]
eigen_values = (pca@stdev)^2
variance_explained = eigen_values / sum(eigen_values)
print(variance_explained)
print(sum(variance_explained[1:10]))
print(sum(variance_explained[1:20]))
print(sum(variance_explained[1:30]))
# 20 PCs have 0.949 variance explained 

# UMAP
seurat_integrated = RunUMAP(seurat_integrated, reduction = "pca", dims = 1:20)
DimPlot(seurat_integrated, reduction = "umap", group.by = "SampleID", label = TRUE) +
  labs(title = "UMAP Plot After Batch Correction", subtitle = "Colored by Sample ID")

# Plot UMAP split by sample
DimPlot(seurat_integrated, split.by = "SampleID")

# Find neighbors and clusters
seurat_integrated = FindNeighbors(seurat_integrated, dims = 1:20, k.param = 10, verbose = FALSE)
seurat_integrated = FindClusters(seurat_integrated, resolution = 0.8, verbose = FALSE)

# Plot UMAP split by sample
DimPlot(seurat_integrated, reduction = "umap", split.by = "SampleID", label = TRUE, cols = NULL) + 
  theme_minimal() + 
  labs(title = "UMAP Plot After Integration and Clustering", subtitle = "Colored by Sample ID") +
  NoLegend()

# Calculate count table
count_table = table(seurat_integrated@meta.data$seurat_clusters, seurat_integrated@meta.data$SampleID)
print(count_table)

# Define custom function to plot integrated clusters
plot_integrated_clusters = function(seurat_obj)
{
  ggplot(seurat_obj@meta.data, aes(x = seurat_clusters, fill = SampleID)) +
    geom_bar(position = "stack") +
    labs(title = "Cluster Distribution Among Samples", x = "Cluster", y = "Number of Cells") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_discrete(name = "Sample ID")
}

# Plot distribution among clusters
plot_integrated_clusters(seurat_integrated)

# Differentially expressed genes in the integrated dataset

# Markers for every cluster
seurat_integrated.markers = FindAllMarkers(seurat_integrated, only.pos = TRUE)
seurat_integrated.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1)

top_genes_per_cluster = seurat_integrated.markers %>%
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

seurat_integrated = SetIdent(seurat_integrated, value = "seurat_clusters")
seurat_integrated$seurat_clusters = as.numeric(factor(seurat_integrated$seurat_clusters))

DoHeatmap(seurat_integrated, features=top5$gene, group.by="seurat_clusters") + NoLegend() + theme(text = element_text(size = 10))

# Assign cell type to clusters

# (1) SingleR annotation

# Get reference datasets from celldex package
monaco.ref = celldex::MonacoImmuneData()

# Set default assay to "RNA" for further processing
DefaultAssay(seurat_integrated) = "RNA"

# Join layers
seurat_integrated = JoinLayers(seurat_integrated, overwrite = TRUE)

# Extract data
data.input = GetAssayData(seurat_integrated, assay = "RNA", slot = "data")

# Convert to SingleCellExperiment
sce = SingleCellExperiment(assays = list(counts = data.input))

# Run SingleR for cell type annotation
monaco.main = SingleR(test = sce, assay.type.test = 1, ref = monaco.ref, labels = monaco.ref$label.main)
monaco.fine = SingleR(test = sce, assay.type.test = 1, ref = monaco.ref, labels = monaco.ref$label.fine)

# Summary of general cell type annotations
table(monaco.main$pruned.labels)

# Summary of finer cell type annotations
table(monaco.fine$pruned.labels)

# Add annotations to the Seurat object metadata
seurat_integrated@meta.data$monaco.main = monaco.main$pruned.labels
seurat_integrated@meta.data$monaco.fine = monaco.fine$pruned.labels

# Visualize the fine-grained annotations
seurat_integrated = SetIdent(seurat_integrated, value = "monaco.main")
DimPlot(seurat_integrated, label = TRUE, repel = TRUE, label.size = 4) + labs(title = "Annotation using SingleR")

# (2) SCTYPE annotation

source(paste0(sctype_dir, "gene_sets_prepare.R"))
source(paste0(sctype_dir, "sctype_score_.R"))

# Built in cell-type-specific gene set database
gs_list = gene_sets_prepare(paste0(sctype_dir, "ScTypeDB_short.xlsx"), "Immune system")

# Load scaled data, when using Seurat, use "RNA" slot with 'scale.data' by default
scRNAseqData = GetAssayData(object = seurat_integrated[["RNA"]], slot = "scale.data")

# Assign cell types
es.max = sctype_score(scRNAseqData = scRNAseqData, scaled = TRUE, gs = gs_list$gs_positive, gs2 = gs_list$gs_negative)

# Merge by cluster
cL_resutls = do.call("rbind", lapply(unique(seurat_integrated@meta.data$seurat_clusters), function(cl)
{
  es.max.cl = sort(rowSums(es.max[ ,rownames(seurat_integrated@meta.data[seurat_integrated@meta.data$seurat_clusters==cl, ])]), decreasing = !0)
  head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl, ncells = sum(seurat_integrated@meta.data$seurat_clusters==cl)), 10)
}
))
sctype_scores = cL_resutls %>% 
  group_by(cluster) %>% 
  top_n(n = 1, wt = scores)  

# Set low-confident (low ScType score) clusters to "unknown"
sctype_scores$type[as.numeric(as.character(sctype_scores$scores)) < sctype_scores$ncells/4] = "Unknown"
print(sctype_scores[,1:3])

# Add annotations to the Seurat object metadata
seurat_integrated@meta.data$sctype_classification = ""
for(j in unique(sctype_scores$cluster))
{
  cl_type = sctype_scores[sctype_scores$cluster==j,]
  seurat_integrated@meta.data$sctype_classification[seurat_integrated@meta.data$seurat_clusters == j] = as.character(cl_type$type[1])
}

# Visualise the annotations
seurat_integrated = SetIdent(seurat_integrated, value = "sctype_classification")
DimPlot(seurat_integrated, label = TRUE, repel = TRUE, label.size = 4) + labs(title = "Annotation using ScType")

# Save as RDS file
saveRDS(seurat_integrated, file=paste0(rds_dir, "seurat_obj_cca_annotated.rds"))
seurat_integrated = readRDS(file=paste0(rds_dir, "seurat_obj_cca_annotated.rds"))
