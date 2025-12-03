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

    def plot_ml_metrics(self, results, plot_filename=None):
        import matplotlib.pyplot as plt

        # Extract model names and accuracies
        models = list(results.keys())
        accuracies = [results[model]['accuracy_mean'] * 100 for model in models]
        accuracy_stds = [results[model]['accuracy_std'] * 100 for model in models]
        losses = [results[model]['loss_mean'] for model in models]
        loss_stds = [results[model]['loss_std'] for model in models]

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Accuracy Plot
        ax1.bar(models, accuracies, yerr=accuracy_stds, capsize=10,
                color=['blue', 'green', 'red', 'purple'])
        ax1.set_title('Machine Learning Models - Accuracy')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 1, f'{v:.2f}%', ha='center')

        # Loss Plot
        ax2.bar(models, losses, yerr=loss_stds, capsize=10,
                color=['blue', 'green', 'red', 'purple'])
        ax2.set_title('Machine Learning Models - Loss')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Loss')
        for i, v in enumerate(losses):
            ax2.text(i, v + 0.0005, f'{v:.4f}', ha='center')

        plt.tight_layout()

        # Save the plot
        if plot_filename is None:
            plot_filename = "snpeff_plots_ml_models.png"
        plt.savefig(plot_filename)
        plt.close()

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
        if "sample_id" in data.columns:
            data.drop("sample_id", axis=1, inplace=True)

        data.fillna(0, inplace=True)

        # Convert to smaller dtypes where possible
        for col in data.columns:
            if col != "status" and data[col].dtype == np.float64:
                data[col] = data[col].astype(np.float32)

        x = data.drop("status", axis=1)
        y = data["status"]
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
        self.plot_ml_metrics(results)

        return results

    def pytorch_svm(self, x, y, kf):
        # Custom PyTorch SVM implementation for GPU acceleration

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

            # Convert to PyTorch tensors and ensure they on the same device
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
            for epoch in range(100):  # Adjust epochs
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

    def train_pytorch_model(self, model_class, x_train, y_train, x_test, y_test, model_args=None, epochs=30,
                            batch_size=64):
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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   pin_memory=(device.type == 'cpu'), num_workers=0)

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

        # Tracking metrics
        train_losses = []
        val_losses = []
        val_accuracies = []

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

            # Record training loss
            train_loss_avg = total_loss / len(train_loader)
            train_losses.append(train_loss_avg)

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

                # Record validation loss and accuracy
                val_losses.append(val_loss.item())
                val_pred = val_outputs.cpu().numpy()
                val_pred_binary = (val_pred > 0.5).astype(int).flatten()
                val_accuracy = accuracy_score(y_test, val_pred_binary)
                val_accuracies.append(val_accuracy)

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

            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss_avg:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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

        # Plotting
        import matplotlib.pyplot as plt
        model_name = model_class.__name__.lower().replace('model', '')
        plot_filename = f"snpeff_plots_{model_name}.png"
        plt.figure(figsize=(10, 6))

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot validation accuracy
        ax1.plot(val_accuracies, label='Validation Accuracy', marker='o', color='blue')
        ax1.set_title(f'{model_class.__name__} Validation Accuracy Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True)
        ax1.legend()

        # Plot losses
        ax2.plot(train_losses, label='Training Loss', marker='o', color='red')
        ax2.plot(val_losses, label='Validation Loss', marker='o', color='green')
        ax2.set_title(f'{model_class.__name__} Losses Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        ax2.legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()

        # Clean up memory
        del x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return model, accuracy

    def dl_models(self):
        # Use copy with memory efficient dtypes
        data = self.df.copy()
        data.drop("sample_id", axis=1, inplace=True)
        data.fillna(0, inplace=True)

        # Convert to smaller dtypes
        for col in data.columns:
            if col != "status" and data[col].dtype == np.float64:
                data[col] = data[col].astype(np.float32)

        x = data.drop("status", axis=1).values.astype(np.float32)
        y = data["status"].values.astype(np.float32)

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

    # df = pd.read_csv("/mnt/data/shyam/anushka/testing/created/autos/final_df.csv") # If the csv file is ready, directly use this
    # df = pd.read_csv("/mnt/data/shyam/anushka/testing/created/autos/latent_with_asd_labels.csv", on_bad_lines='skip')
    df = pd.read_csv("/mnt/data/shyam/anushka/testing/created/autos/vep_50kv2/latent_with_asd_labels.csv", on_bad_lines="skip")
    print(df.shape)
    # ML models
    print("\nML models will start now: ")
    ml_start_time = time.time()
    classify = ClassificationModels(df)
    ml_results = classify.ml_models()
    ml_time = time.time() - ml_start_time
    print(f"ML models completed in {ml_time} seconds, yayyyy!!! ")


    # DL models
    print("\nDL models will start npw: ")
    dl_start_time = time.time()
    dl_results = classify.dl_models()
    dl_time = time.time() - dl_start_time
    print(f"DL models completed in {dl_time} seconds, yayyyyyyyyyy!!! ")

    # Summary for ML
    print("\nML Summary:")
    for model, metrics in ml_results.items():
        print(f"{model}: Accuracy: {metrics['accuracy_mean'] * 100:.2f}% ± {metrics['accuracy_std'] * 100:.2f}%")

    # Summary for DL
    print("\nDL Summary:")
    for model, metrics in dl_results.items():
        print(f"{model}: Accuracy: {metrics['accuracy_mean'] * 100:.2f}% ± {metrics['accuracy_std'] * 100:.2f}%")

    # Summary for total time taken at each step
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time} seconds ")
    print(f"VCF files processing: {vcf_time} seconds, {vcf_time / total_time * 100}% of the time was taken for this")
    print(f"Running ML models: {ml_time} seconds, {ml_time / total_time * 100}% of the time was taken for this")
    print(f"Running DL models: {dl_time}seconds, {dl_time / total_time * 100}% of the time was taken for this")


if __name__ == "__main__":
    main()
