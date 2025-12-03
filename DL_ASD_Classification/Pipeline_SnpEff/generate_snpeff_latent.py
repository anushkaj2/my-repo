import csv
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
import gc
import re
from scipy import sparse
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import multiprocessing
from contextlib import nullcontext


# Set the start method to SPAWN as default for linux is fork
multiprocessing.set_start_method("spawn", force=True)
# Ignore useless warnings
warnings.filterwarnings("ignore")


class CSVDataset(Dataset):
    def __init__(self, noisy_data, original_data, device="cpu"):
        # Creating tensors on CPU as I/O operations are more efficient on the CPU rather than GPU
        # Then use DataLoader to move to GPU for training in Dataset batches
        # This makes code clean and better
        self.device = device
        self.noisy_tensor = torch.tensor(noisy_data.values, dtype=torch.float32)
        self.original_tensor = torch.tensor(original_data.values, dtype=torch.float32)
        # Only move to device i.e. GPU, if specified (just some error handling)
        # To avoid runtime errors:
        if device != "cpu":
            try:
                self.noisy_tensor = self.noisy_tensor.to(device)
                self.original_tensor = self.original_tensor.to(device)
            except RuntimeError:
                print("Tensors are too large for GPU memory, keeping them on CPU. ")
                self.device = "cpu"

    def __len__(self):
        return len(self.noisy_tensor)

    def __getitem__(self, idx):
        # This method will access the individual tensors by index and return tuple of (input, target) tensors
        return self.noisy_tensor[idx], self.original_tensor[idx]


class OptimizedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super(OptimizedAutoencoder, self).__init__()
        # Define a new class OptimisedAutoencoder which inherits from PyTorch's base class nn.Module
        hidden_dim = 512

        # Encoder: 2 Linear layers with LayerNorm and ReLU activation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm as it can work with small batches too
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)  # Changed from BatchNorm
        )

        # Decoder: 2 Linear layers with LayerNorm and ReLU activation, and output layer is Sigmoid
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # Range of output: (0, 1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class VCFprocessor:
    def __init__(self, input_dir, asd_genes, asd_data_file, output_csv, batch_size=100, num_workers=8):
        self.input_dir = input_dir
        # Load the first column in asd_genes, convert to a list, then to a set for speedy look up O(1) time complexity
        self.asd_genes = set(pd.read_csv(asd_genes, header=None).iloc[:, 0].tolist())
        # To prevent Pandas from assigning data types by itself and mess things up, declare them here
        self.asd_data = pd.read_csv(asd_data_file, sep=",", dtype={"subject_sp_id": str, "asd": "category"})
        self.output_csv = output_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        # To increase performance
        self.gene_pattern = re.compile(r"ANN=.*?\|.*?\|.*?\|(.*?)\|")
        self.variant_pattern = re.compile(r"ANN=.*?\|(.*?)\|")
        self.impact_pattern = re.compile(r"ANN=.*?\|.*?\|(.*?)\|")
        # Take only unique subject_sp_id, label encode asd column, zip them, turn to dictionary for faster look up (Faster then Pandas indexing)
        self.asd_dict = dict(zip(self.asd_data["subject_sp_id"], self.asd_data["asd"].map({"True": 1, "False": 0})))
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def extract_gene_name(self, info):
        matches = self.gene_pattern.findall(info)
        return ",".join(set(matches)) if matches else "Unknown"

    def extract_variant_type(self, info):
        matches = self.variant_pattern.findall(info)
        return ",".join(set(matches)) if matches else "Unknown"

    def extract_impact(self, info):
        matches = self.impact_pattern.findall(info)
        return ",".join(set(matches)) if matches else "Unknown"

    def extract_zygosity(self, format_field):
        # This should be renamed to "extract_genotype"
        return format_field[:format_field.find(":")]

    def process_vcf_file(self, filename):
        # Process a single VCF file
        lines_skipped = 0
        gvcf_file = os.path.join(self.input_dir, filename)
        file_name = os.path.splitext(filename)[0]


        # Use context manager to create 1 MB chunks, extracted data at each lien will be stored in a list, then appended to a df
        data = []
        try:
            # Open each gvcf file 1MB at a time, as loading the whole file can slow the process
            with open(gvcf_file, "r", buffering=1024 * 1024) as f:
                for line in f:
                    # If a line in VCF file is corrupted, it starts with x00, so we should skip it
                    if line.startswith("#") or "\x00" in line:
                        if "\x00" in line:
                            lines_skipped += 1
                        continue

                    columns = line.strip().split("\t")
                    if len(columns) < 10:  # Ensure we have enough columns, if not, then VCF file was not annotated properly
                        continue

                    ref = columns[3]  # REF column for reference allele
                    alt = columns[4]  # ALT column for alternate allele
                    info = columns[7]  # INFO field for information on variant, impact
                    format_field = columns[9]  # FORMAT field for genotype

                    # Extract required fields only for those genes, that are related to ASD
                    gene_name = self.extract_gene_name(info)
                    if not any(gene in self.asd_genes for gene in gene_name.split(",")):
                        continue

                    variant_type = self.extract_variant_type(info)
                    impact = self.extract_impact(info)
                    zygosity = self.extract_zygosity(format_field)

                    data.append({"gene_name": gene_name, "ref": ref, "alt": alt,
                                 "variant_type": variant_type, "impact": impact, "zygosity": zygosity})
        # Error handling
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None


        if lines_skipped > 0:
            print(f"Lines skipped in {filename}: {lines_skipped} (due to null values being present)")
        if not data:
            print(f"No data found in {filename}, skipping")
            return None
        df = pd.DataFrame(data)
        # If df is empty/very small, the df is not good, mostly comes from a corrupted VCF file
        if df.empty or df.memory_usage(deep=True).sum() / 1024 <= 10:
            print(f"Insufficient data in {filename}, skipping")
            return None


        # One hot encode the df: Original df (Input for the AutoEncoder)
        columns_to_encode = ["ref", "alt", "impact", "zygosity"]
        one_hot_encoded_df = pd.DataFrame()
        one_hot_encoded_df["gene_name"] = df["gene_name"]
        # Process categorical columns more efficiently
        for col in columns_to_encode:
            # Convert to categorical first for memory efficiency
            df[col] = pd.Categorical(df[col])
            one_hot = pd.get_dummies(df[col], prefix=col, dtype=np.int8)
            one_hot_encoded_df = pd.concat([one_hot_encoded_df, one_hot], axis=1)
        # Delete the original df, to save memory as it is not needed now
        del df


        # Add masking noise: Noisy df (Input for the AutoEncoder)
        noisy_df = one_hot_encoded_df.copy()
        numeric_cols = [col for col in noisy_df.columns if col != "gene_name" and pd.api.types.is_numeric_dtype(noisy_df[col])]
        # Create noise mask using GPU (via PyTorch) or CPU (via NumPy): Flip 10% of the values
        if self.device == "cuda":
            noise_tensor = torch.rand(len(noisy_df), len(numeric_cols), device=self.device)
            noise_mask = (noise_tensor < 0.1).cpu().numpy()  # Move back to CPU for pandas
            del noise_tensor
            torch.cuda.empty_cache()  # Free GPU memory
        else:
            noise_mask = (np.random.rand(len(noisy_df), len(numeric_cols)) < 0.1)
        # Apply noise mask by setting values to 0 where they were originally 1 and the noise mask is True.
        for i, col in enumerate(numeric_cols):
            # Only modify rows where value is 1 and noise mask is True
            noisy_df.loc[(noisy_df[col] == 1) & noise_mask[:, i], col] = 0


        # Remove the gene_name column from both the dfs
        noisy_data = noisy_df.drop("gene_name", axis=1)
        original_data = one_hot_encoded_df.drop("gene_name", axis=1)

        # Pass the two dfs to train an AutoEncoder
        return self.train_autoencoder(noisy_data, original_data, file_name)


    def train_autoencoder(self, noisy_data, original_data, file_name):
        # Is CUDA available?
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f"Using this device: {device}")


        # Create dataset using our good old custom dataset class, ONLY USING CPU FOR THIS TASK
        dataset = CSVDataset(noisy_data, original_data, device='cpu')
        # Not using a large batch here for training, we do not want to run out of memory
        data_size = len(dataset)
        batch_size = min(32, max(1, data_size // 8))
        # Create a PyTorch DataLoader, and transfer each batch from CPU to GPU
        # Pinned memory does this transfer faster than normal memory
        # TRIAL NEEDED (Multiprocessing): Try num_workers = 4, to spawn 4 subprocesses to load data faster to GPU
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda, num_workers=0)
        # These dfs are not needed anymore
        del noisy_data, original_data


        # Input dim = Number of one hot encoded columns in each file (~102 columns)
        input_dim = dataset.noisy_tensor.shape[1]
        model = OptimizedAutoencoder(input_dim=input_dim, latent_dim=128).to(device)
        print(f"Model moved to: {device}")
        # Update all of model's weights using a variant of Adam called AMSGrad (this one helps with convergence)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
        # Loss is calculated by: (predicted-original)^2
        criterion = nn.MSELoss()
        # If validation loss dos not improve, decrease lr by half (adjust lr when stuck/plateaus)
        # We want to minimise the lr optimiser, by 0.5, wait for 2 epochs before doing that
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        # For mixed precision as some computations can do with float16 rather than float32 to save GPU memory
        # GradScaler will scale float16 values up
        scaler = torch.cuda.amp.GradScaler() if use_cuda else None


        # Training Mode ON
        model.train()
        # Early stopping is defined to precent overfitting
        best_loss = float("inf")
        patience_counter = 0
        max_patience = 5
        best_model_state = None

        for epoch in range(25):
            total_loss = 0
            batch_count = 0

            for noisy_batch, original_batch in dataloader:
                # Now move each batch of tensors to GPU
                noisy_batch = noisy_batch.to(device)
                original_batch = original_batch.to(device)
                # As BatchNormalisation cannot take batch of just size 1, we skip such batches
                if noisy_batch.size(0) <= 1:
                    continue
                batch_count += 1
                # Reset gradient to clear gradients from previous batches
                optimizer.zero_grad()
                # If CUDA is found use that or else turn to CPU
                if use_cuda:
                    # Autocast to train model with mixed precision
                    with torch.cuda.amp.autocast():
                        reconstructed, _ = model(noisy_batch)
                        loss = criterion(reconstructed, original_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    reconstructed, _ = model(noisy_batch)
                    loss = criterion(reconstructed, original_batch)
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()

            # Find the loss and avoid division by 0
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
                # Update lr based on loss
                scheduler.step(avg_loss)
                # Early stopping if model does not imrpve for the last 5 epochs
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"Early stopping at epoch: {epoch + 1}")
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                        break
            else:
                print(f"Epoch {epoch + 1}: No valid batches processed")


        # Evaluation Mode ON
        model.eval()
        with torch.no_grad():
            # Extract latent representations and move it to the CPU
            all_latent = []
            for i in range(0, len(dataset), batch_size):
                batch_end = min(i + batch_size, len(dataset))
                batch_noisy = dataset.noisy_tensor[i:batch_end].to(device)
                try:
                    batch_latent = model.encoder(batch_noisy)
                    all_latent.append(batch_latent.cpu())
                except RuntimeError as e:
                    print(f"Error encoding batch: {i}-{batch_end}: {e}")
                    continue
            # Are latent vectors produced?
            if not all_latent:
                print("No valid latent vectors could be generated")
                return None
            # Concatenate latent vector from all batches using mean
            all_latent_tensor = torch.cat(all_latent)
            latent_vector = all_latent_tensor.mean(dim=0, keepdim=True).cpu().numpy()
        # Free GPU memory of cache
        del model, dataset, dataloader
        if use_cuda:
            torch.cuda.empty_cache()


        # Create DataFrame of Latent Representations
        latent_df = pd.DataFrame(latent_vector)
        latent_df.columns = [f"latent_dim_{i + 1}" for i in range(latent_df.shape[1])]
        latent_df.insert(0, "file_name", file_name)
        return latent_df


    def process_file_batch(self, filenames):
        # Process multiple files and then combine the results
        batch_results = []
        for filename in filenames:
            result = self.process_vcf_file(filename)
            if result is not None:
                batch_results.append(result)
        # Combine results if any are produced
        if batch_results:
            return pd.concat(batch_results, ignore_index=True)
        return None


    def vcf2csv(self):
        aggregated_df = pd.DataFrame()
        # Get all VCF files from the input directory
        vcf_files = [f for f in os.listdir(self.input_dir) if f.endswith(".gvcf")]
        # Divide the files into batches for batch processing
        total_files = len(vcf_files)
        print(f"Processing {total_files} files in batches of {self.batch_size}")
        # Use multiprocessing to parallelise file processing
        file_batches = [vcf_files[i:i + self.batch_size]
                        for i in range(0, len(vcf_files), self.batch_size)]

        # Loop through each batch of files
        for batch_idx, batch in enumerate(file_batches):
            print(f"Processing batch {batch_idx + 1}/{len(file_batches)}")
            start_time = time.time()

            # Using multiprocessing library to parallel process each batch
            # Num workers is given as 16
            with ProcessPoolExecutor(max_workers=min(self.num_workers, len(batch))) as executor:
                # Using tqdm for progress bar
                futures = {executor.submit(self.process_vcf_file, filename): filename for filename in batch}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Files"):
                    filename = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            aggregated_df = pd.concat([aggregated_df, result], ignore_index=True)
                    except Exception as exc:
                        print(f"File {filename} generated an exception: {exc}")

            # Force garbage collection after each batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch_time = time.time() - start_time
            print(f"Batch {batch_idx + 1} completed in {batch_time:.2f}s ({batch_time / len(batch):.2f}s per file)")

            # Check if DataFrame is empty or missing 'file_name' column
            if aggregated_df.empty:
                print("No data was processed successfully. Cannot continue.")
                return pd.DataFrame()

            if 'file_name' not in aggregated_df.columns:
                print("Error: 'file_name' column missing from processed data.")
                print("Available columns:", aggregated_df.columns.tolist())
                return aggregated_df

        # Add target labels efficiently
        final_updated_data = aggregated_df.copy()
        final_updated_data["target"] = pd.NA  # Use pandas NA for initial values

        # Apply mapping function for efficiency
        final_updated_data["target"] = final_updated_data["file_name"].map(self.asd_dict)

        # Remove rows with no target
        final_updated_data = final_updated_data.dropna(subset=["target"])

        # Convert target to integer
        final_updated_data["target"] = final_updated_data["target"].astype(int)

        # Save to .csv
        final_updated_data.to_csv(self.output_csv, index=False)
        print(f"Processed final csv file: {self.output_csv}")
        print(f"Final dataframe shape: {final_updated_data.shape}")

        return final_updated_data


# Helper context manager for conditional code execution
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): pass


class LSTMModel(nn.Module):
    def __init__(self, input_dim, lstm_units=64):  # Increased units for A100
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_units,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(lstm_units, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        x = self.dropout(lstm_out)
        x = self.relu(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x

class GRUModel(nn.Module):
    def __init__(self, input_dim, gru_units=64):  # Increased units for A100
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=gru_units,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(gru_units, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        # Take the output of the last time step
        gru_out = gru_out[:, -1, :]
        x = self.dropout(gru_out)
        x = self.relu(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x

class FCNNModel(nn.Module):
    def __init__(self, input_dim, hidden_units=128):  # Increased units for A100
        super(FCNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        # x = self.fc2(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64):  # Increased size for A100
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, input_dim, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,  # Increased heads for A100
            dim_feedforward=128,  # Increased for A100
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model * input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        x = self.embedding(x)  # [batch, seq_len, d_model]
        x = x + self.pos_encoder  # Add positional encoding
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        x = x.reshape(x.shape[0], -1)  # Flatten: [batch, seq_len * d_model]
        x = self.sigmoid(self.fc(x))  # [batch, 1]
        # x = self.fc2(x)
        return x

class ClassificationModels:
    def __init__(self, df):
        self.df = df
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def calculate_metrics(self, model, x, y, kf):
        accuracy_scores = []
        loss_scores = []

        for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            loss = log_loss(y_test, y_pred_proba)
            accuracy_scores.append(accuracy)
            loss_scores.append(loss)

            print(f"Fold {fold}: {model.__class__.__name__}: Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")

        print(
            f"{model.__class__.__name__}: Mean Accuracy: {np.mean(accuracy_scores) * 100:.2f}%, SD: {np.std(accuracy_scores) * 100:.2f}%")
        print(f"{model.__class__.__name__}: Mean Loss: {np.mean(loss_scores):.4f}, SD: {np.std(loss_scores):.4f}\n")

        return {
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'loss_mean': np.mean(loss_scores),
            'loss_std': np.std(loss_scores)
        }

    def ml_models(self):
        # Use copy with memory efficient dtypes
        data = self.df.copy()

        # Check if file_name column exists before dropping
        if "file_name" in data.columns:
            data.drop("file_name", axis=1, inplace=True)

        data.fillna(0, inplace=True)

        # Convert to smaller dtypes where possible
        for col in data.columns:
            if col != "target" and data[col].dtype == np.float64:
                data[col] = data[col].astype(np.float32)

        x = data.drop("target", axis=1)
        y = data["target"]
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        results = {}

        # Logistic Regression with optimized solver for GPU
        model = LogisticRegression(solver='saga', max_iter=200, n_jobs=-1)
        results['logistic_regression'] = self.calculate_metrics(model, x, y, kf)
        del model
        gc.collect()

        # Random Forest with GPU-accelerated prediction if using RAPIDS
        model = RandomForestClassifier(
            n_estimators=200,  # Increased for A100 capability
            max_depth=15,  # Increased for A100 capability
            n_jobs=-1,
            random_state=42
        )
        results['random_forest'] = self.calculate_metrics(model, x, y, kf)
        del model
        gc.collect()

        # XGBoost with GPU acceleration
        model = XGBClassifier(
            tree_method='gpu_hist',  # Use GPU acceleration
            predictor='gpu_predictor',  # Use GPU for prediction
            learning_rate=0.1,
            max_depth=8,
            n_estimators=200,
            random_state=42
        )
        results['xgboost'] = self.calculate_metrics(model, x, y, kf)
        del model
        gc.collect()

        # SVM - Use GPU if available via PyTorch implementation
        if self.device.type == 'cuda':
            # For larger datasets, we'll use a custom PyTorch SVM implementation
            results['svm'] = self.pytorch_svm(x, y, kf)
        else:
            # Fall back to scikit-learn SVM
            model = SVC(kernel="rbf", C=1.0, probability=True)
            results['svm'] = self.calculate_metrics(model, x, y, kf)
            del model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def pytorch_svm(self, x, y, kf):
        """Custom PyTorch SVM implementation for GPU acceleration"""

        class TorchSVM(nn.Module):
            def __init__(self, input_dim):
                super(TorchSVM, self).__init__()
                self.linear = nn.Linear(input_dim, 1)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        device = self.device
        accuracy_scores = []
        loss_scores = []

        for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
            x_train, x_test = x.iloc[train_index].values, x.iloc[test_index].values
            y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

            # Convert to PyTorch tensors and ensure they're on the same device
            x_train_tensor = torch.FloatTensor(x_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
            x_test_tensor = torch.FloatTensor(x_test).to(device)
            y_test_tensor = torch.FloatTensor(y_test).to(device)

            # Initialize model and explicitly move to device
            model = TorchSVM(x_train.shape[1]).to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Train model
            model.train()
            for epoch in range(100):  # Adjust epochs as needed
                optimizer.zero_grad()
                outputs = model(x_train_tensor)  # Should be on GPU
                loss = criterion(outputs, y_train_tensor)  # Both tensors should be on GPU
                loss.backward()
                optimizer.step()

            # Evaluate model
            model.eval()
            with torch.no_grad():
                # Ensure all tensors are on the same device
                y_pred = model(x_test_tensor).cpu().numpy()  # Move to CPU before converting to numpy
                y_pred_binary = (y_pred > 0.5).astype(int).flatten()

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred_binary)
            # Calculate log loss with clipping to avoid infinity
            y_pred_proba = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = log_loss(y_test, y_pred_proba)

            accuracy_scores.append(accuracy)
            loss_scores.append(loss)

            print(f"Fold {fold}: PyTorch SVM: Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")

            # Clean up GPU memory
            del model, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        print(
            f"PyTorch SVM: Mean Accuracy: {np.mean(accuracy_scores) * 100:.2f}%, SD: {np.std(accuracy_scores) * 100:.2f}%")
        print(f"PyTorch SVM: Mean Loss: {np.mean(loss_scores):.4f}, SD: {np.std(loss_scores):.4f}\n")

        return {
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'loss_mean': np.mean(loss_scores),
            'loss_std': np.std(loss_scores)
        }

    def train_pytorch_model(self, model_class, x_train, y_train, x_test, y_test, model_args=None, epochs=30, batch_size=64):
        # This is a function to train PyTorch models using GPU acceleration
        if model_args is None:
            model_args = {}

        device = self.device

        # For RNN models, reshape the input
        is_rnn = model_class.__name__ in ["LSTMModel", "GRUModel", "TransformerModel"]
        if is_rnn:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # Convert to PyTorch tensors to ensure all tensors go to the same device
        x_train_tensor = torch.FloatTensor(x_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        x_test_tensor = torch.FloatTensor(x_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)

        # Create datasets and data loaders for efficient GPU training
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor.unsqueeze(1))
        # Only use pin_memory for CPU, Set num_workers = 0 for GPU to avoid memory sharing issues
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cpu'), num_workers=0)

        # Initialize model and explicitly move to the device
        input_dim = x_train.shape[1]
        model = model_class(input_dim, **model_args).to(device)

        # Ensure all model parameters are on the GPU
        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device)

        # Use mixed precision for faster training
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        # Loss and optimizer
        # criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Train the model
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch_x, batch_y in train_loader:
                # Ensure batches are on correct device (should be redundant but safe)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()

                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                # Ensure test tensors are on the correct device (redundant but safe)
                x_test_tensor = x_test_tensor.to(device)
                y_test_tensor = y_test_tensor.to(device)

                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        val_outputs = model(x_test_tensor)
                        val_loss = criterion(val_outputs, y_test_tensor.unsqueeze(1))
                else:
                    val_outputs = model(x_test_tensor)
                    val_loss = criterion(val_outputs, y_test_tensor.unsqueeze(1))

            # Update learning rate
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # Final evaluation
        model.eval()
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    y_pred = model(x_test_tensor).cpu().numpy()  # Move to CPU before numpy conversion
            else:
                y_pred = model(x_test_tensor).cpu().numpy()

            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_test, y_pred_binary)

        # Clean up memory
        del x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return model, accuracy

    def dl_models(self):
        # Use copy with memory efficient dtypes
        data = self.df.copy()
        data.drop("file_name", axis=1, inplace=True)
        data.fillna(0, inplace=True)

        # Convert to smaller dtypes
        for col in data.columns:
            if col != "target" and data[col].dtype == np.float64:
                data[col] = data[col].astype(np.float32)

        x = data.drop("target", axis=1).values.astype(np.float32)
        y = data["target"].values.astype(np.float32)

        # Use smaller number of folds with A100 to get faster results
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        model_classes = {
            'lstm': LSTMModel,
            'gru': GRUModel,
            'fcnn': FCNNModel,
            'transformer': TransformerModel
        }

        model_args = {
            'lstm': {'lstm_units': 128},  # Increased for A100
            'gru': {'gru_units': 128},  # Increased for A100
            'fcnn': {'hidden_units': 256},  # Increased for A100
            'transformer': {'d_model': 128}  # Increased for A100
        }

        results = {}

        for model_name, model_class in model_classes.items():
            print(f"\nTraining {model_name.upper()} model...")
            fold_accuracies = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(x), 1):
                print(f"Fold {fold}/{kf.n_splits}")
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train model
                _, accuracy = self.train_pytorch_model(
                    model_class,
                    x_train, y_train,
                    x_test, y_test,
                    model_args=model_args[model_name],
                    epochs=30,
                    batch_size=128 if model_name == 'fcnn' else 64  # Larger batch size for FCNN
                )

                fold_accuracies.append(accuracy)
                print(f"Fold {fold} accuracy: {accuracy:.4f}")

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Calculate statistics
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            print(f"{model_name.upper()}: Mean Accuracy: {mean_acc * 100:.2f}%, SD: {std_acc * 100:.2f}%")

            results[model_name] = {
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc
            }

        return results

def main():
    # Set PyTorch to use A100 efficiently
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    else:
        print("CUDA not available, CPU will be used. (This message is not supposed to show up if running on the GPU).")

    # All the input files/directories needed by the code are defined are
    input_dir = "/mnt/data/shyam/anushka/testing/created/trial2/data_files"
    asd_genes = "/mnt/data/shyam/anushka/testing/created/trial2/sfari_gene_selected.csv"
    asd_data_file = "/mnt/data/shyam/anushka/testing/created/trial2/asd_labels.csv"
    output_csv = "/mnt/data/shyam/anushka/testing/created/trial2/final_df.csv"

    # This is when the code starts doing its job
    start_time = time.time()

    print("VCF files will be processed now to generate the final_df.csv file: ")
    try:
        # Parallel processing with 16 workers and 100 vcf files would be processed at the same time
        vcf2df = VCFprocessor(input_dir, asd_genes, asd_data_file, output_csv, batch_size=100, num_workers=16)
        df = vcf2df.vcf2csv()
        # This is how long it took the code to process all the VCF files
        vcf_time = time.time() - start_time
        print(f"VCF files were processed in: {vcf_time} seconds!!! Lets goooo!! ")

        # Just some steps to ensure the df is doing good.
        required_columns = ["target"]
        if "file_name" not in df.columns:
            print(f"Yo, file_name column not found in the df.")
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            print(f"Missing required columns: {missing_required}. Execution of this code will be stopped now.")
            return

    except Exception as e:
        # More information on the error, if error errors
        print(f"Error in execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
