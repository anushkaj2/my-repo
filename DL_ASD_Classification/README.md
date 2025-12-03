
1. Project Title: Autoencoder-Based Latent Embeddings for Whole-Exome Sequencing in ASD


2. Project Overview: 
Autism Spectrum Disorder (ASD) diagnosis relies mainly on behavioural assessments, which are subjective and often delayed. This project investigates whether whole-exome sequencing data can be utilised to learn latent genomic representations using denoising autoencoders and contrastive learning models. Two pipelines (SnpEff and VEP-based) were created to annotate variants, preprocess VCF files, learn embeddings, and classify ASD status using ML/DL methods.


3. Repository Structure

project_root/
│
├── Pipeline_SnpEff/
│   ├── README.md
│   ├── generate_snpeff_latent.py
│   ├── train_classical_models.py
│   └── contrastive_learning.py
│
├── Pipeline_VEP/
│   ├── README.md
│   ├── filter_variants_sfari.py
│   ├── annotate_with_vep.py
│   ├── generate_latent_representations.py
│   ├── train_classical_models.py
│   └── contrastive_learning.py
│
└── environment.yml


4. Methods Implemented

a. Representation Learning
b. Denoising Autoencoders (PyTorch)
c. Contractive regularization (for robustness)
d. Per-file latent embedding extraction
e. Contrastive learning using NT-Xent loss
f. Classification Models
g. Logistic Regression
h. Random Forest
i. XGBoost (GPU-enabled)
j. SVM (CPU/GPU)
k. Fully-connected NN
l. LSTM / GRU
m. Transformer Encoder


5. Data Requirements

a. Raw gVCF variant files (not provided in repo)
b. Annotation databases (SnpEff or Ensembl VEP)
c, Metadata CSV containing:
sample_id | ASD_status

Note: Raw genomic data cannot be shared due to privacy



6. Results Summary

The results show that both ML and DL models perform similarly in classifying ASD using autoencoder-generated features. The Transformer model achieved ~60% accuracy with low SD, indicating stable performance. These findings highlight the potential of latent representations in genomic classification. With further optimization this approach could evolve into a clinically useful, AI-driven diagnostic tool for ASD.


7. Motivation & Impact

Using 50,000 .gVCF files from SPARK, key genetic features were extracted and processed with SnpEff, latent representations were extracted using autoencoders, enabling models to achieve ~60% accuracy with low variance. Future improvements include refining feature selection and experimenting with new architectures. This is to develop a user-friendly, web-based tool that processes VCF files and outputs latent genomic features, aiding clinicians and researchers in early ASD diagnosis and analysis.


8. Links to Pipelines

SnpEff Pipeline: https://github.com/anushkaj2/my-repo/tree/main/DL_ASD_Classification/Pipeline_SnpEff
VEP Pipeline: https://github.com/anushkaj2/my-repo/tree/main/DL_ASD_Classification/Pipeline_VEP



VEP v1 Pipeline → Pipeline_VEP_v1/README.md

VEP v2 Pipeline → Pipeline_VEP_v2/README.md
