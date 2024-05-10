Problem: Identify genes that might be associated with developing Autism
Dataset: The dataset being used here is GSE42133 from Gene Expression Omnibus that has mRNA Leukocyte gene expression values.
Data extraction: Grep commands were run on the LInux command line and python was used to get relevant data out. Additionally, R's Bioconductor was used to map probe_id's to genes. 
Models: (1) Supervised models (with 10 folds) were run.
        (2) First 146 Principal Components from PCA were used and passed into run_supervised_models function.
        (3) SHAP was used to get the top 30 genes that may contribute towards developing Autism
Results: Mean model performance with 10 folds after getting appropriate genes from SHAP is 87% with a low standard deviation.
Conclusion: Further analysis should be done to understand how the identified genes may contribute to this.
