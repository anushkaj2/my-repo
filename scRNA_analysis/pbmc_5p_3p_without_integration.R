# DATA INTEGRATION: Plot UMAP without Integrating the Datasets

library(Seurat)
library(tidyverse)
library(Matrix)
library(ggplot2)

data_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\dataset\\"
plots_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\plots\\"
rds_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\saved_rds\\"

# Load RDS file
seurat_obj = readRDS(file = paste0(rds_dir, "seurat_obj_loaded.rds"))

# QC 
seurat_obj[["percent.mt"]] = PercentageFeatureSet(seurat_obj, pattern = "^MT-")
VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
# Feature scatter plots to compare different QC metrics
plot1 = FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 = FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot1 + plot2
seurat_obj = subset(seurat_obj, subset = nFeature_RNA>300 & nFeature_RNA<2500 & percent.mt<5)
VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

# Normalisation 
seurat_obj = NormalizeData(seurat_obj, normalization.method="LogNormalize", scale.factor=10000)

# Feature Selection
seurat_obj= FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000) 

# Identify the 10 most highly variable genes
top10 = head(VariableFeatures(seurat_obj), 10)
print(top10)

# Plot top10 variable features with labels
LabelPoints(plot = VariableFeaturePlot(seurat_obj), points = top10, repel = TRUE)

# Scale data
all.genes = rownames(seurat_obj)
seurat_obj = ScaleData(seurat_obj, features=all.genes)

# PCA (Linear dimensionality reduction)
# To run PCA fast, select only the top 500 variable features, calculate the first 20 PCs and approximate the calculations
seurat_obj = RunPCA(seurat_obj, features=head(VariableFeatures(seurat_obj), 500), approx=TRUE, npcs=20) 
print(seurat_obj[["pca"]], dims = 1:5, nfeatures = 5)

# Cluster
seurat_obj = FindNeighbors(seurat_obj, dims = 1:20)
seurat_obj = FindClusters(seurat_obj, resolution = 0.8)

# UMAP (Non linear dimensionality reduction)
seurat_obj = RunUMAP(seurat_obj, dims=1:20)
DimPlot(seurat_obj, reduction = "umap", group.by = "SampleID", label = TRUE) 

# Save as RDS file
saveRDS(seurat_obj, file=paste0(rds_dir, "seurat_obj_without_integration.rds"))
seurat_obj = readRDS(file=paste0(rds_dir, "seurat_obj_without_integration.rds"))
