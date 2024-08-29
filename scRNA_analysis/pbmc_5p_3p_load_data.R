# DATA INEGRATION: Load Data

library(Seurat)
library(tidyverse)
library(Matrix)

data_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\dataset\\"
rds_dir = "C:\\Users\\Dell\\Desktop\\IBAB\\strand_internship\\pbmc_5p_3p\\saved_rds\\"

# Read the data and create Seurat objects
matrix_3p = Read10X_h5(paste0(data_dir, "3p_pbmc10k_filt.h5"), use.names = T)

# 5’ dataset has other assays: VDJ data
matrix_5p = Read10X_h5(paste0(data_dir, "5p_pbmc10k_filt.h5"),use.names = T)$`Gene Expression`


seurat_3p = CreateSeuratObject(matrix_3p, project = "pbmc10k_3p")
seurat_3p@meta.data$SampleID = "pbmc_3p"
seurat_5p = CreateSeuratObject(matrix_5p, project = "pbmc10k_5p")
seurat_5p@meta.data$SampleID = "pbmc_5p"

# Remove matrices to save memory
rm(matrix_3p)
rm(matrix_5p)

# Create list of Seurat objects
seurat_list = list(seurat_3p, seurat_5p)

# Merge into one big Seurat object
seurat_obj = merge(x=seurat_list[[1]], y=seurat_list[2:length(seurat_list)])
print(seurat_obj)

# Save as RDS file
saveRDS(seurat_obj, file=paste0(rds_dir, "seurat_obj_loaded.rds"))