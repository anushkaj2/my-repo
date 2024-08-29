# DATA INEGRATION: Harmony

library(Seurat)
library(tidyverse)
library(Matrix)
library(ggplot2)
library(harmony)
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

# Run PCA and UMAP for visualization before Harmony integration
seurat_obj[["percent.mt"]] = PercentageFeatureSet(seurat_obj, pattern = "^MT-")
seurat_obj = subset(seurat_obj, subset = nFeature_RNA>500 & nFeature_RNA<4000 & percent.mt<5)
seurat_obj = NormalizeData(seurat_obj, verbose = F)
seurat_obj = FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000, verbose = F)
seurat_obj = ScaleData(seurat_obj, verbose = F)
seurat_obj = RunPCA(seurat_obj, npcs = 30, verbose = F)
# Find how many PCs should be included for downstream analysis
pca = seurat_obj[["pca"]]
eigen_values = (pca@stdev)^2
variance_explained = eigen_values / sum(eigen_values)
print(variance_explained)
print(sum(variance_explained[1:10]))
print(sum(variance_explained[1:20]))
print(sum(variance_explained[1:30]))
# 30 PCs have 1.0 variance explained 
seurat_obj = RunUMAP(seurat_obj, reduction = "pca", dims = 1:30, verbose = F)

# Plot UMAP before integration
DimPlot(seurat_obj, reduction = "umap", group.by = "SampleID") + labs(title = "Before Integration: pbmc_3p and pbmc_5p Cells")

# Run Harmony integration
seurat_obj = RunHarmony(object = seurat_obj, group.by = "SampleID", dims.use = 1:30, plot_convergence=T)
DimPlot(object = seurat_obj, reduction = "harmony", pt.size = .1, group.by = "SampleID")

# Run UMAP and do clustering
seurat_obj = seurat_obj %>% 
  RunUMAP(reduction = "harmony", dims = 1:30, verbose = F) %>% 
  FindNeighbors(reduction = "harmony", k.param = 10, dims = 1:30) %>% 
  FindClusters(resolution = 0.8) %>% 
  identity()

# Plot after harmony
seurat_obj = SetIdent(seurat_obj, value = "SampleID")
DimPlot(seurat_obj, reduction = "umap") + labs(title = "After Harmony: pbmc_3p and pbmc_5p Cells")

# Separate the two samples
DimPlot(seurat_obj, reduction = "umap", group.by = "SampleID", pt.size = .1, split.by = 'SampleID') + NoLegend()

# With clusters
seurat_obj =  SetIdent(seurat_obj, value = "seurat_clusters")
DimPlot(seurat_obj, label = T) + NoLegend()

# Count table
count_table = table(seurat_obj@meta.data$seurat_clusters, seurat_obj@meta.data$SampleID)
print(count_table)

# Define custom function to plot integrated clusters
plot_integrated_clusters <- function(seurat_obj) {
  ggplot(seurat_obj@meta.data, aes(x = seurat_clusters, fill = SampleID)) +
    geom_bar(position = "stack") +
    labs(title = "Cluster Distribution Among Samples", x = "Cluster", y = "Number of Cells") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_discrete(name = "Sample ID")
}
plot_integrated_clusters(seurat_obj)

# Find markers for all clusters
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

DoHeatmap(seurat_obj, features=top5$gene) + NoLegend() + theme(text = element_text(size = 10))

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
DimPlot(seurat_obj, label = TRUE, repel = TRUE, label.size = 3) + NoLegend()

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
saveRDS(seurat_obj, file=paste0(rds_dir, "seurat_obj_harmony_annotated.rds"))
seurat_obj =readRDS(file=paste0(rds_dir, "seurat_obj_harmony_annotated.rds"))
