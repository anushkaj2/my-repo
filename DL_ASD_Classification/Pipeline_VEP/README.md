Pipeline: VEP → Autoencoder → Classification

This folder contains the complete VEP-based representation-learning pipeline for converting raw gVCF / VCF files into latent genomic embeddings and using them for ASD classification through classical ML and contrastive-learning approaches.

1. filter_variants.py
Purpose:
Filters raw .gvcf / .vcf files to keep only variants belonging to the SFARI gene set. This step reduces file size and focuses the downstream model on ASD-relevant genes.
What it does:
a. Loads the gene list from a CSV and generates a .txt list used for filtering
b. Scans each VCF line and retains lines containing any SFARI gene
c. Outputs *_filtered.vcf files


2. run_vep_annotation.py
Purpose:
Annotates the filtered VCF files using Ensembl VEP with dbNSFP features (CADD, GERP++, PROVEAN, gnomAD, phyloP, phastCons, etc.).
What it does:
a. Takes all *_filtered.vcf files
b. Runs VEP using multiple CPU processes
c. Injects rich functional scores and conservation metrics
d. Produces *_vep_filt.vcf annotated files


3. generate_latent_representations.py
Purpose:
Convert VEP-annotated files into machine-learnable embeddings via a denoising autoencoder.
Key steps:
a. Parses VEP .vcf into a clean DataFrame
b. Handles numeric & categorical annotation fields
c. Feature engineering (e.g., conservation averages, rare variant flags, log-distance, damage flags)
d. Scales numeric data
e. Builds a denoising autoencoder (VCFAutoencoder)
f. Trains using CPU/GPU with mixed precision
g. Extracts a 512-dim latent vector per sample
Outputs:
a. latent_features_final.csv
b. latent_with_asd_labels.csv (after mapping sample → ASD label)


4. train_classical_models.py
Purpose:
Train interpretable ML models on autoencoder embeddings.
Models trained include:
a. Logistic Regression
b. Random Forest
c. XGBoost (GPU-accelerated)
d. SVM (CPU/GPU)
e. LSTM, GRU, FCNN, Transformer (PyTorch)
What it does:
a. Performs 10-fold cross-validation
b. Computes accuracy, log-loss (mean ± SD)
c. Generates performance plots
d. Saves logs/plots for comparison


5. contrastive_learning.py
Purpose:
Implements a self-supervised contrastive learning pipeline (NT-Xent loss) for genomic representation learning.
Core components:
a. Data augmentation functions (noise/scale/flip)
b. Encoder + projection head
c. Contrastive loss computation
d. Validation monitoring (accuracy, val loss, error)
e. Downstream evaluation via Logistic Regression


Inputs:
1. Raw gVCF files
2. SFARI gene list
3. ASD labels file
4. Output directory for saving Filtered VCFs, Annotated VCFs, Latent representations, ML model plots, Contrastive learning plots


How to run:
python3 filter_variants.py
CUDA_VISIBLE_DEVICES=1 python3 run_vep_annotation.py
CUDA_VISIBLE_DEVICES=1 python3 generate_latent_representations.py
CUDA_VISIBLE_DEVICES=1 python3 train_classical_models.py
CUDA_VISIBLE_DEVICES=1 python3 contrastive_learning.py


Outputs
1. latent_features_final.csv
2. latent_with_asd_labels.csv
3. ml_accuracy_loss_plots.png (or similar)
4. contrastive_training_accuracy.png
5. contrastive_loss_curves.png


