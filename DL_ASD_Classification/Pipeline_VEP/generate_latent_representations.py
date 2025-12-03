import os
import torch
import pandas as pd
import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import resource

# Set higher file limit for the current process
# This helps but might not be enough depending on the system configuration
def increase_file_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Try to set to hard limit first
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"Increased file limit from {soft} to {hard}")
    except ValueError:
        # If that fails, try a reasonable high number
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
            print(f"Increased file limit from {soft} to 4096")
        except ValueError:
            print(f"Could not increase file limit. Current limit: {soft}")
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]

# Restrict to least occupied GPU (GPU 3)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Enable cuDNN auto-tuner for optimal performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class VCFPreprocess:
    def __init__(self, input_path):
        self.df = None
        self.autoencoder_ready_df = None
        if os.path.isdir(input_path):
            all_files = glob.glob(os.path.join(input_path, "*"))
            self.vcf_files = sorted(f for f in all_files if os.path.isfile(f) and f.lower().endswith(".vcf"))
            if not self.vcf_files:
                raise FileNotFoundError(f"No .vcf files found in directory: {input_path}")
            self.current_vcf = self.vcf_files[0]
            print(f"Found {len(self.vcf_files)} VCF(s); defaulting to: {os.path.basename(self.current_vcf)}")
        elif os.path.isfile(input_path):
            self.vcf_files = [input_path]
            self.current_vcf = input_path
            print(f"Selected single VCF: {os.path.basename(self.current_vcf)}")
        else:
            raise FileNotFoundError(f"Path not found: {input_path}")

    def list_vcf_files(self):
        for i, file in enumerate(self.vcf_files):
            print(f"{i}: {os.path.basename(file)}")
        return self.vcf_files

    def select_vcf_file(self, index_or_name):
        if isinstance(index_or_name, int) and 0 <= index_or_name < len(self.vcf_files):
            self.current_vcf = self.vcf_files[index_or_name]
        elif isinstance(index_or_name, str):
            matches = [f for f in self.vcf_files if index_or_name in f]
            if matches:
                self.current_vcf = matches[0]
            else:
                print(f"No file matching '{index_or_name}' found")
                return False
        else:
            print("Invalid index or name")
            return False
        print(f"Selected: {os.path.basename(self.current_vcf)}")
        self.df = None
        self.autoencoder_ready_df = None
        return True

    def load_vcf_data(self):
        if not self.current_vcf:
            print("No VCF file selected")
            return None
        data = []
        header = []
        try:
            with open(self.current_vcf, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        if line.startswith('#Uploaded_variation'):
                            header = line[1:].strip().split('\t')
                        continue
                    fields = line.strip().split('\t')
                    # Replace '-' placeholders with NaN
                    fields = [np.nan if f == '-' else f for f in fields]
                    data.append(fields)
            if not header:
                print("Warning: No header found, using default header.")
                header = ["Uploaded_variation", "Location", "Allele", "Gene", "Feature", "Feature_type",
                          "Consequence", "cDNA_position", "CDS_position", "Protein_position", "Amino_acids",
                          "Codons", "Existing_variation", "IMPACT", "DISTANCE", "STRAND", "FLAGS", "APPRIS",
                          "CADD_phred", "GERP++_RS", "Interpro_domain", "MutationAssessor_score", "PROVEAN_score",
                          "Reliability_index", "SIFT4G_score", "SIFT_score", "aapos", "bStatistic",
                          "codon_degeneracy", "codonpos", "gnomAD4.1_joint_AF", "phastCons100way_vertebrate",
                          "phastCons17way_primate", "phyloP100way_vertebrate", "phyloP17way_primate"]
            self.df = pd.DataFrame(data, columns=header)
            # Convert columns to numeric where applicable
            numeric_cols = [c for c in self.df.columns if any(
                pat in c for pat in ["cDNA_position", "CDS_position", "Protein_position", "DISTANCE", "CADD_phred", "GERP", "MutationAssessor_score", "PROVEAN_score", "SIFT", "bStatistic", "AF", "phastCons", "phyloP"]
            )]
            self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            print(f"Loaded {len(self.df)} variants from {os.path.basename(self.current_vcf)}")
            return self.df
        except Exception as e:
            print(f"Error loading VCF data: {str(e)}")
            return None

    def preprocess_for_autoencoder(self):
        """Preprocess VCF data for autoencoder with vectorized operations for speed"""
        if self.df is None:
            print("No data loaded. Please load VCF data first.")
            return None

        try:
            # Make a copy using float16 to save memory
            df = self.df.copy()

            # Pre-calculate all needed statistics at once
            numeric_stats = {}
            for col in ["cDNA_position", "CDS_position", "Protein_position", "DISTANCE"]:
                if col in df.columns:
                    numeric_stats[col] = {
                        'median': df[col].median(),
                        'max': df[col].max()
                    }

            # Fill missing values by column-wise strategy all at once
            fill_values = {}
            for col in df.columns:
                if col == "cDNA_position" and col in numeric_stats:
                    fill_values[col] = numeric_stats[col]['median']
                elif col == "CDS_position" and col in numeric_stats:
                    fill_values[col] = numeric_stats[col]['median']
                elif col == "Protein_position" and col in numeric_stats:
                    fill_values[col] = numeric_stats[col]['median']
                elif col == "DISTANCE" and col in numeric_stats:
                    fill_values[col] = numeric_stats[col]['max']
                elif any(pattern in col for pattern in ["CADD_phred", "GERP++_RS", "MutationAssessor_score",
                                                        "PROVEAN_score", "SIFT_score", "bStatistic", "gnomAD",
                                                        "phastCons", "phyloP"]):
                    fill_values[col] = 0

            # Apply fillna all at once
            if fill_values:
                df.fillna(fill_values, inplace=True)

            # Create missing indicators in a vectorized way
            missing_indicators = {f'missing_{col}': df[col].isna().astype('int8')
                                  for col in fill_values.keys() if col in df.columns}
            if missing_indicators:
                missing_df = pd.DataFrame(missing_indicators)
                df = pd.concat([df, missing_df], axis=1)
                del missing_df  # Free memory

            # === Feature Engineering - vectorized operations ===
            # Combined impact
            if all(c in df.columns for c in ["CADD_phred", "GERP++_RS", "SIFT_score"]):
                df["Combined_impact"] = (df["CADD_phred"] * df["GERP++_RS"] * (1 - df["SIFT_score"])).astype('float16')

            # Conservation scores
            if all(c in df.columns for c in ["phastCons100way_vertebrate", "phastCons17way_primate"]):
                df["avg_phastCons"] = df[["phastCons100way_vertebrate", "phastCons17way_primate"]].mean(axis=1).astype(
                    'float16')
            if all(c in df.columns for c in ["phyloP100way_vertebrate", "phyloP17way_primate"]):
                df["avg_phyloP"] = df[["phyloP100way_vertebrate", "phyloP17way_primate"]].mean(axis=1).astype('float16')

            # Frequency-related features
            gnomad_cols = [col for col in df.columns if "gnomAD" in col and col.endswith("_AF")]
            if gnomad_cols:
                # Calculate all frequency metrics at once
                df_gnomad = df[gnomad_cols]
                df["AF_mean"] = df_gnomad.mean(axis=1).astype('float16')
                df["AF_max"] = df_gnomad.max(axis=1).astype('float16')
                df["AF_std"] = df_gnomad.std(axis=1).astype('float16')
                df["is_rare_variant"] = (df["AF_max"] < 0.01).astype('int8')
                del df_gnomad  # Free memory

            # Binary damage flags - vectorized with int8 dtype
            binary_flags = {
                "is_CADD_high": (df["CADD_phred"] > 20).astype('int8') if "CADD_phred" in df.columns else pd.Series(
                    [0] * len(df), dtype='int8'),
                "is_PROVEAN_damaging": (df["PROVEAN_score"] < -2.5).astype(
                    'int8') if "PROVEAN_score" in df.columns else pd.Series([0] * len(df), dtype='int8'),
                "is_SIFT_damaging": (df["SIFT_score"] < 0.05).astype(
                    'int8') if "SIFT_score" in df.columns else pd.Series([0] * len(df), dtype='int8'),
                "is_MA_high": (df["MutationAssessor_score"] > 2.0).astype(
                    'int8') if "MutationAssessor_score" in df.columns else pd.Series([0] * len(df), dtype='int8')
            }
            for col, values in binary_flags.items():
                df[col] = values

            # Log transform DISTANCE
            if "DISTANCE" in df.columns:
                df["log_DISTANCE"] = np.log1p(df["DISTANCE"]).astype('float16')

            # === Final preprocessing for autoencoder ===
            # Get numeric columns only
            autoencoder_input_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            X = df[autoencoder_input_cols].fillna(0)  # Extra safety

            # Normalize with float16 precision for memory efficiency
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X).astype('float16')

            self.autoencoder_ready_df = pd.DataFrame(X_scaled, columns=autoencoder_input_cols)

            print(f"Prepared {self.autoencoder_ready_df.shape[1]} features for autoencoder input.")
            return self.autoencoder_ready_df

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


class VCFDataset(Dataset):
    def __init__(self, dataframe, noise_std=0.05):
        # Convert to torch tensor with optimized precision
        self.data = torch.tensor(dataframe.values, dtype=torch.float16)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # Generate noise for denoising autoencoder
        noise = torch.randn_like(x) * self.noise_std
        return x + noise, x


class VCFAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super(VCFAutoencoder, self).__init__()

        # Encoder: Linear → LayerNorm → ReLU → Linear → LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Decoder: Linear → LayerNorm → ReLU → Linear
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x):
        # Returns the latent representation
        return self.encoder(x)


def train_autoencoder(
    model,
    dataloader,
    epochs: int = 25,
    learning_rate: float = 1e-4,
    device=None,
    memory_fraction: float = 0.9
):
    """Train the denoising autoencoder with optional contractive loss."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.backends.cudnn.benchmark = True
        print(f"Using {memory_fraction * 100:.0f}% of available GPU memory")

    print(f"Training on: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Use proper GradScaler for mixed precision training
    scaler = None
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    # For annealing if used
    try:
        initial_noise = dataloader.dataset.noise_std
    except AttributeError:
        initial_noise = 0.0

    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_noisy, batch_original in progress_bar:
            batch_noisy = batch_noisy.to(device, non_blocking=True)
            batch_noisy = batch_noisy.detach()  # ensure leaf
            batch_noisy.requires_grad_(True)  # enable grad
            batch_original = batch_original.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward + compute losses under AMP
            if scaler:
                with torch.cuda.amp.autocast():
                    reconstructed, latent = model(batch_noisy)
                    recon_loss = criterion(reconstructed, batch_original)

                    # Contractive penalty
                    grads = torch.autograd.grad(
                        outputs=latent,
                        inputs=batch_noisy,
                        grad_outputs=torch.ones_like(latent),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    lam = 1e-4  # strength of contractive regularization
                    contractive_loss = lam * torch.sum(grads.pow(2))

                    loss = recon_loss + contractive_loss

                scaler.scale(loss).backward()
                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstructed, latent = model(batch_noisy)
                recon_loss = criterion(reconstructed, batch_original)

                grads = torch.autograd.grad(
                    outputs=latent,
                    inputs=batch_noisy,
                    grad_outputs=torch.ones_like(latent),
                    create_graph=True,
                    retain_graph=True
                )[0]
                lam = 1e-4
                contractive_loss = lam * torch.sum(grads.pow(2))
                loss = recon_loss + contractive_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # cleanup
            del batch_noisy, batch_original, reconstructed, latent, grads
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")

        # early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().detach() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                break

    return model


def add_asd_labels(latent_df, labels_path):
    """Add ASD status labels to the latent representation DataFrame"""
    asd_data = pd.read_csv(labels_path, dtype={'asd': str})
    latent_df["status"] = ""
    rows_to_drop = []

    for idx, row in latent_df.iterrows():
        sample_name = row["sample_id"]

        matching = asd_data[asd_data["subject_sp_id"] == sample_name]
        if not matching.empty:
            val = matching.iloc[0]["asd"]
            latent_df.at[idx, "status"] = 1 if val == "True" else 0
        else:
            rows_to_drop.append(idx)
            print(f"No matching ASD label for sample_id={sample_name!r}")

    if rows_to_drop:
        latent_df.drop(index=rows_to_drop, inplace=True)
    return latent_df


def process_file(file_path: str, vcf_dir: str, device_str: str, worker_id: int):
    """Process a single VCF file with added worker_id for tracking and proper resource management"""
    try:
        # Import torch in each worker process to avoid scoping issues
        import torch
        import os
        from torch.utils.data import DataLoader
        
        # Use restricted CUDA device to prevent memory exhaustion
        if device_str == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use GPU 3 as in original script
        
        device = torch.device(device_str)
        
        print(f"Worker {worker_id} starting file: {os.path.basename(file_path)}")
        
        # Create a new preprocessor specific to this process
        preprocessor = VCFPreprocess(os.path.join(vcf_dir, file_path))
        preprocessor.load_vcf_data()
        autoencoder_ready_df = preprocessor.preprocess_for_autoencoder()

        if autoencoder_ready_df is None or autoencoder_ready_df.empty:
            print(f"Worker {worker_id}: Skipping empty file {file_path}")
            return None

        # Create dataset with reduced worker count to minimize file handles
        dataset = VCFDataset(autoencoder_ready_df)
        batch_size = min(1024, len(dataset))
        
        # Key change: reduce num_workers to prevent too many open files
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,  # REDUCED from 4 to 1
            persistent_workers=False  # No persistent workers
        )

        input_dim = autoencoder_ready_df.shape[1]
        model = VCFAutoencoder(input_dim=input_dim, latent_dim=512)
        
        # Train with explicit device
        trained_model = train_autoencoder(model, dataloader, device=device, epochs=25)

        # Get latent representation
        with torch.no_grad():
            inputs = torch.tensor(autoencoder_ready_df.values, dtype=torch.float32).to(device)
            latent_vec = trained_model.encode(inputs).mean(dim=0).cpu().numpy()

        # Extract sample ID
        sample_id = os.path.basename(file_path).replace("_vep_filt.vcf", "")
        
        # Ensure we clean up resources
        del dataset, dataloader, model, trained_model, inputs
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        print(f"Worker {worker_id} completed file: {os.path.basename(file_path)}")
        return np.append(latent_vec, sample_id)

    except Exception as e:
        print(f"Worker {worker_id} error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_parallel_processing(vcf_files, vcf_dir, output_dir, max_workers=4):
    """Run processing in parallel with controlled worker count and proper resource management"""
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

    # Get only the file names, not full paths
    vcf_filenames = [os.path.basename(f) for f in vcf_files]
    
    # Use spawn context for clean process isolation
    ctx = multiprocessing.get_context('spawn')
    
    # Process files in batches to control resource usage
    batch_size = 10  # Process 10 files at a time
    for batch_idx in range(0, len(vcf_filenames), batch_size):
        batch_files = vcf_filenames[batch_idx:batch_idx + batch_size]
        batch_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(process_file, file_path, vcf_dir, device_str, i): file_path
                for i, file_path in enumerate(batch_files)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc=f"Processing batch {batch_idx//batch_size + 1}/{(len(vcf_filenames)-1)//batch_size + 1}"):
                res = future.result()
                if res is not None:
                    batch_results.append(res)
        
        # Append batch results to overall results
        results.extend(batch_results)
        print(f"Completed batch {batch_idx//batch_size + 1}, processed {len(batch_results)} files successfully.")
        
        # Force garbage collection between batches
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def integrated_asd_pipeline(vcf_dir, asd_labels_path, output_dir, use_gpu=True):
    """Main pipeline with improved resource management"""
    try:
        # Increase file limit
        file_limit = increase_file_limit()
        print(f"Running with file descriptor limit: {file_limit}")
        
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Using device: {device}")

        # Initialize preprocessor just to get file list
        preprocessor = VCFPreprocess(vcf_dir)
        vcf_files = preprocessor.list_vcf_files()
        print(f"Found {len(vcf_files)} VCF files to process.")

        # Calculate appropriate max_workers based on system and file limits
        max_cores = multiprocessing.cpu_count()
        # Adjust max_workers based on file limit - each process needs resources
        suggested_workers = min(max_cores // 2, file_limit // 100)  
        max_workers = max(1, min(4, suggested_workers))  # Between 1 and 4
        print(f"Using {max_workers} worker processes (system has {max_cores} cores)")

        # Process files in managed parallel
        all_latents = run_parallel_processing(vcf_files, vcf_dir, output_dir, max_workers=max_workers)

        if not all_latents:
            print("No data processed successfully.")
            return
            
        print(f"Successfully processed {len(all_latents)} files.")
        
        # Convert results to DataFrame
        latent_df = pd.DataFrame(all_latents, columns=[f"dim_{i}" for i in range(512)] + ["sample_id"])
        latent_df.to_csv(os.path.join(output_dir, "latent_features_final.csv"), index=False)
        print(f"Saved latent features to {os.path.join(output_dir, 'latent_features_final.csv')}")

        # Add ASD labels
        final_df = add_asd_labels(latent_df, asd_labels_path)
        final_df.to_csv(os.path.join(output_dir, "latent_with_asd_labels.csv"), index=False)
        print(f"Pipeline complete. Final dataset with {len(final_df)} samples saved to {os.path.join(output_dir, 'latent_with_asd_labels.csv')}")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set multiprocessing start method before any pool creation
    multiprocessing.set_start_method('spawn', force=True)
    
    vcf_dir = "/mnt/data/shyam/anushka/testing/created/annotated_vep"
    asd_labels_path = "/mnt/data/shyam/anushka/testing/created/trial2/asd_labels.csv"
    output_dir = "/mnt/data/shyam/anushka/testing/created/autos/vep_50kv1"
    
    integrated_asd_pipeline(
        vcf_dir,
        asd_labels_path,
        output_dir,
        use_gpu=True
    )
