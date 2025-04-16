# import csv
# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, log_loss
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, SimpleRNN, Input, LayerNormalization, MultiHeadAttention, \
#     Flatten, Softmax, Reshape
# from tensorflow.keras.optimizers import Adam
#
# class CSVDataset(Dataset):
#     def __init__(self, noisy_data, original_data):
#         # Accept dataframes directly instead of filenames
#         self.noisy_data = noisy_data
#         self.original_data = original_data
#         self.noisy_tensor = torch.tensor(self.noisy_data.values, dtype=torch.float32)
#         self.original_tensor = torch.tensor(self.original_data.values, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.noisy_tensor)
#
#     def __getitem__(self, idx):
#         return self.noisy_tensor[idx], self.original_tensor[idx]
#
#
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(nn.Linear(input_dim, 128),
#                                      nn.ReLU(),
#                                      nn.Linear(128, latent_dim))
#         self.decoder = nn.Sequential(nn.Linear(latent_dim, 128),
#                                      nn.ReLU(),
#                                      nn.Linear(128, input_dim),
#                                      nn.Sigmoid())
#
#     def forward(self, x):
#         latent = self.encoder(x)
#         reconstructed = self.decoder(latent)
#         return reconstructed, latent
#
#
# class VCFprocessor:
#     def __init__(self, input_dir, asd_genes, asd_data_file, output_csv=None):
#         self.input_dir = input_dir
#         self.asd_genes = asd_genes
#         self.asd_data_file = asd_data_file
#         self.output_csv = output_csv if output_csv else os.path.join(input_dir, "final_processed_data.csv")
#
#     def extract_gene_name(self, info):
#         # Extract gene name(s) from INFO field (ANN annotation)
#         if "ANN=" in info:
#             annotations = info.split("ANN=")[1].split(",")
#             gene_names = set()
#             for annotation in annotations:
#                 parts = annotation.split("|")
#                 if len(parts) > 3:  # Check if gene name exists in the annotation
#                     gene_names.add(parts[3])
#             return ",".join(gene_names)
#         return "Unknown"
#
#     def extract_variant_type(self, info):
#         # Extract variant type from INFO field (ANN annotation)
#         if "ANN=" in info:
#             annotations = info.split("ANN=")[1].split(",")
#             variant_types = set()
#             for annotation in annotations:
#                 parts = annotation.split("|")
#                 if len(parts) > 1:  # Check if variant type exists in the annotation
#                     variant_types.add(parts[1])
#             return ",".join(variant_types)
#         return "Unknown"
#
#     def extract_impact(self, info):
#         # Extract impact from INFO field
#         if "ANN=" in info:
#             annotations = info.split("ANN=")[1].split(",")
#             impacts = set()
#             for annotation in annotations:
#                 parts = annotation.split("|")
#                 if len(parts) > 2:  # Check if impact exists in the annotation
#                     impacts.add(parts[2])
#             return ",".join(impacts)
#         return "Unknown"
#
#     def extract_zygosity(self, format_field):
#         # Extract zygosity from FORMAT field
#         return format_field.split(":")[0]
#
#     def vcf2csv(self):
#         df = pd.read_csv(self.asd_genes, header=None)
#         gene_list = df.iloc[:, 0].tolist()
#         aggregated_df = pd.DataFrame()
#         asd_data = pd.read_csv(self.asd_data_file, sep=',', dtype={'asd': str})
#         final_final_df = pd.DataFrame()
#
#         for filename in os.listdir(self.input_dir):
#             if filename.endswith(".gvcf"):
#                 print(filename)
#                 lines_skipped = 0
#                 gvcf_file = os.path.join(self.input_dir, filename)
#                 file_name = os.path.splitext(filename)[0]  # Extract file name without extension
#
#                 # Create a temporary CSV file path
#                 csv_file = os.path.join(self.input_dir, f"{file_name}_temp.csv")
#
#                 with open(gvcf_file, "r") as f:
#                     reader = f.readlines()
#
#                 with open(csv_file, "w", newline='') as csvfile:
#                     fieldnames = ["gene_name", "ref", "alt", "variant_type", "impact", "zygosity"]
#                     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                     writer.writeheader()
#
#                     data = []
#
#                     for line in reader:
#                         if line.startswith("#") or "\x00" in line:
#                             if "\x00" in line:
#                                 lines_skipped = lines_skipped + 1
#                             continue
#
#                         columns = line.strip().split("\t")
#                         if len(columns) < 10:  # Ensure we have enough columns
#                             continue
#
#                         ref = columns[3]  # REF column
#                         alt = columns[4]  # ALT column
#                         info = columns[7]  # INFO field
#                         format_field = columns[9]  # FORMAT field for zygosity
#
#                         # Extract required fields
#                         gene_name = self.extract_gene_name(info)
#                         variant_type = self.extract_variant_type(info)
#                         impact = self.extract_impact(info)
#                         zygosity = self.extract_zygosity(format_field)
#
#                         data.append({"gene_name": gene_name, "ref": ref, "alt": alt,
#                                      "variant_type": variant_type, "impact": impact,
#                                      "zygosity": zygosity})
#
#                     if lines_skipped > 0:
#                         print(f"Lines skipped in {filename}: {lines_skipped} (due to null values being present)")
#
#                     df = pd.DataFrame(data)
#
#                 def gene_matches(row):
#                     genes = str(row).split(",")
#                     return any(gene in gene_list for gene in genes)
#
#                 # Skip if df is empty
#                 if df.empty:
#                     print(f"No data found in {filename}, skipping")
#                     continue
#
#                 filtered_df = df[df["gene_name"].apply(gene_matches)]
#
#                 # Skip if filtered_df is empty
#                 if filtered_df.empty:
#                     print(f"No matching genes found in {filename}, skipping")
#                     continue
#
#                 if filtered_df.memory_usage(deep=True).sum() / 1024 > 10:
#                     columns_to_encode = ["ref", "alt", "impact", "zygosity"]
#                     one_hot_encoded_df = pd.DataFrame()
#                     one_hot_encoded_df["gene_name"] = filtered_df["gene_name"]
#
#                     for col in columns_to_encode:
#                         one_hot = pd.get_dummies(filtered_df[col], prefix=col, dtype=int)
#                         one_hot_encoded_df = pd.concat([one_hot_encoded_df, one_hot], axis=1)
#                 else:
#                     print(f"Skipping file due to small size: {filename}")
#                     continue
#
#                 # Make a noisy version for the autoencoder
#                 noisy_df = one_hot_encoded_df.copy()
#                 for col in noisy_df.columns:
#                     if col != "gene_name" and noisy_df[col].dtype in [np.int64,
#                                                                       np.float64]:  # Ensure it's numeric (0/1)
#                         mask = (noisy_df[col] == 1) & (np.random.rand(len(noisy_df[col])) < 0.1)  # 10% probability
#                         noisy_df.loc[mask, col] = 0  # Flip 1s to 0s
#
#                 # Remove the gene_name column for the dataset
#                 noisy_data = noisy_df.drop("gene_name", axis=1)
#                 original_data = one_hot_encoded_df.drop("gene_name", axis=1)
#
#                 dataset = CSVDataset(noisy_data, original_data)
#                 dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
#                 # Initialize Autoencoder
#                 input_dim = dataset.noisy_tensor.shape[1]
#                 model = Autoencoder(input_dim=input_dim, latent_dim=32)
#                 optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#                 criterion = nn.MSELoss()
#
#                 def train_autoencoder(model, dataloader, optimizer, criterion, epochs=10):
#                     model.train()
#                     for epoch in range(epochs):
#                         total_loss = 0
#                         for noisy_batch, original_batch in dataloader:
#                             optimizer.zero_grad()
#                             reconstructed, _ = model(noisy_batch)
#                             loss = criterion(reconstructed, original_batch)
#                             loss.backward()
#                             optimizer.step()
#                             total_loss += loss.item()
#                         print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
#
#                 # Train the model
#                 train_autoencoder(model, dataloader, optimizer, criterion, epochs=10)
#
#                 # Extract latent representation for the entire dataset
#                 model.eval()
#                 with torch.no_grad():
#                     latent_vector = model.encoder(dataset.noisy_tensor).mean(dim=0, keepdim=True)
#
#                 # Save latent representations to a DataFrame
#                 latent_df = pd.DataFrame(latent_vector)
#                 latent_df.columns = [f"latent_dim_{i + 1}" for i in range(latent_df.shape[1])]
#                 latent_df.insert(0, "file_name", file_name)
#
#                 # Add to aggregated dataframe
#                 aggregated_df = pd.concat([aggregated_df, latent_df], ignore_index=True)
#
#                 # Clean up temporary file
#                 os.remove(csv_file)
#
#         # Add target labels
#         final_updated_data = aggregated_df.copy()
#         final_updated_data["target"] = ""
#         rows_to_drop = []
#
#         for index, row in final_updated_data.iterrows():
#             sample_name = row["file_name"]
#             matching_row = asd_data.loc[asd_data["subject_sp_id"] == sample_name]
#             if not matching_row.empty:
#                 asd_value = matching_row.iloc[0]["asd"]
#                 if asd_value == "True":
#                     final_updated_data.loc[index, 'target'] = 1
#                 elif asd_value == 'False':
#                     final_updated_data.loc[index, 'target'] = 0
#             else:
#                 rows_to_drop.append(index)
#                 print(f"No matching sample name found in ASD data for {sample_name}")
#
#         final_updated_data.drop(index=rows_to_drop, inplace=True)
#         final_updated_data['target'] = final_updated_data['target'].astype(int)
#         final_updated_data.to_csv(self.output_csv, index=False)
#         print(f"Processed finasl csv file: {self.output_csv}")
#
#         return final_updated_data
#
#
# class ClassificationModels:
#     def __init__(self, df):
#         self.df = df
#
#     def calculate_metrics(self, model, x, y, kf):
#         accuracy_scores = []
#         loss_scores = []
#
#         for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
#             x_train, x_test = x.iloc[train_index], x.iloc[test_index]
#             y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#             model.fit(x_train, y_train)
#             y_pred = model.predict(x_test)
#             y_pred_proba = model.predict_proba(x_test)
#
#             accuracy = accuracy_score(y_test, y_pred)
#             loss = log_loss(y_test, y_pred_proba)
#             accuracy_scores.append(accuracy)
#             loss_scores.append(loss)
#
#             print(f"Fold {fold}: {model.__class__.__name__}: Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")
#
#         print(
#             f"{model.__class__.__name__}: Mean Accuracy: {np.mean(accuracy_scores) * 100:.2f}%, SD: {np.std(accuracy_scores) * 100:.2f}%")
#         print(f"{model.__class__.__name__}: Mean Loss: {np.mean(loss_scores):.4f}, SD: {np.std(loss_scores):.4f}\n")
#
#         return {
#             'accuracy_mean': np.mean(accuracy_scores),
#             'accuracy_std': np.std(accuracy_scores),
#             'loss_mean': np.mean(loss_scores),
#             'loss_std': np.std(loss_scores)
#         }
#
#     def ml_models(self):
#         data = self.df.copy()
#         data.drop("file_name", axis=1, inplace=True)
#         data.fillna(0, inplace=True)
#         x = data.drop("target", axis=1)
#         y = data["target"]
#         kf = KFold(n_splits=10, shuffle=True, random_state=42)
#
#         results = {}
#
#         # Logistic Regression
#         model = LogisticRegression()
#         results['logistic_regression'] = self.calculate_metrics(model, x, y, kf)
#
#         # Random Forest
#         model = RandomForestClassifier(n_estimators=200, random_state=42)
#         results['random_forest'] = self.calculate_metrics(model, x, y, kf)
#
#         # XGBoost
#         model = XGBClassifier()
#         results['xgboost'] = self.calculate_metrics(model, x, y, kf)
#
#         # SVM
#         model = SVC(kernel="linear", C=1.0, probability=True)
#         results['svm'] = self.calculate_metrics(model, x, y, kf)
#
#         return results
#
#     def dl_models(self):
#         def build_lstm_model(input_dim, lstm_units=64, dense_units=32, dropout_rate=0.3):
#             model = Sequential()
#             model.add(LSTM(lstm_units, activation="relu", input_shape=(input_dim, 1)))
#             model.add(Dropout(dropout_rate))
#             model.add(Dense(dense_units, activation="relu"))
#             model.add(Dropout(dropout_rate))
#             model.add(Dense(1, activation="sigmoid"))
#             model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#             return model
#
#         def build_gru_model(input_dim, rnn_units=64, dense_units=32, dropout_rate=0.3):
#             model = Sequential()
#             model.add(GRU(rnn_units, activation="relu", input_shape=(input_dim, 1)))
#             model.add(Dropout(dropout_rate))
#             model.add(Dense(dense_units, activation="relu"))
#             model.add(Dropout(dropout_rate))
#             model.add(Dense(1, activation="sigmoid"))
#             model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#             return model
#
#         def build_fcnn_model(input_dim, dense_units=64, dropout_rate=0.3):
#             model = Sequential()
#             model.add(Dense(dense_units, activation="relu", input_shape=(input_dim,)))
#             model.add(Dropout(dropout_rate))
#             model.add(Dense(dense_units // 2, activation="relu"))
#             model.add(Dropout(dropout_rate))
#             model.add(Dense(1, activation="sigmoid"))
#             model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#             return model
#
#         def build_transformer_model(input_dim):
#             inputs = Input(shape=(input_dim,))
#             x = Reshape((input_dim, 1))(inputs)
#
#             # Transformer block
#             x = Dense(64, activation="relu")(x)
#             x = LayerNormalization()(x)
#             x = Dropout(0.2)(x)
#             attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
#             x = attention_output + x  # Skip connection
#             x = LayerNormalization()(x)
#             x = Dense(32, activation="relu")(x)
#             x = Dropout(0.2)(x)
#             x = Flatten()(x)  # Flatten before the final layer
#             outputs = Dense(1, activation="sigmoid")(x)  # Binary classification
#             model = Model(inputs=inputs, outputs=outputs)
#             model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#             return model
#
#         data = self.df.copy()
#         # Use entire dataset unless it's too large
#         if len(data) > 500:
#             data = data.iloc[:500, :]  # Use first 500 rows if dataset is large
#
#         data.drop("file_name", axis=1, inplace=True)
#         data.fillna(0, inplace=True)
#         x = data.drop("target", axis=1).values
#         y = data["target"].values
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Use 5-fold for DL due to computational intensity
#
#         lstm_accuracies = []
#         gru_accuracies = []
#         fcnn_accuracies = []
#         transformer_accuracies = []
#
#         for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
#             x_train, x_test = x[train_index], x[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#
#             # Reshape for RNN models
#             x_train_rnn = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
#             x_test_rnn = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
#
#             # LSTM Model
#             lstm_model = build_lstm_model(input_dim=x_train.shape[1])
#             history = lstm_model.fit(x_train_rnn, y_train, epochs=30, batch_size=32,
#                                      validation_split=0.2, verbose=0)
#             loss, accuracy = lstm_model.evaluate(x_test_rnn, y_test, verbose=0)
#             print(f"Fold {fold}: LSTM Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#             lstm_accuracies.append(accuracy)
#
#             # GRU Model
#             gru_model = build_gru_model(input_dim=x_train.shape[1])
#             history = gru_model.fit(x_train_rnn, y_train, epochs=30, batch_size=32,
#                                     validation_split=0.2, verbose=0)
#             loss, accuracy = gru_model.evaluate(x_test_rnn, y_test, verbose=0)
#             print(f"Fold {fold}: GRU Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#             gru_accuracies.append(accuracy)
#
#             # FCNN Model
#             fcnn_model = build_fcnn_model(input_dim=x_train.shape[1])
#             history = fcnn_model.fit(x_train, y_train, epochs=30, batch_size=32,
#                                      validation_split=0.2, verbose=0)
#             loss, accuracy = fcnn_model.evaluate(x_test, y_test, verbose=0)
#             print(f"Fold {fold}: FCNN Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#             fcnn_accuracies.append(accuracy)
#
#             # Transformer Model
#             transformer_model = build_transformer_model(input_dim=x_train.shape[1])
#             history = transformer_model.fit(x_train, y_train, epochs=30, batch_size=32,
#                                             validation_split=0.2, verbose=0)
#             loss, accuracy = transformer_model.evaluate(x_test, y_test, verbose=0)
#             print(f"Fold {fold}: Transformer Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#             transformer_accuracies.append(accuracy)
#
#         print(f"LSTM: Mean Accuracy: {np.mean(lstm_accuracies) * 100:.2f}%, SD: {np.std(lstm_accuracies) * 100:.2f}%")
#         print(f"GRU: Mean Accuracy: {np.mean(gru_accuracies) * 100:.2f}%, SD: {np.std(gru_accuracies) * 100:.2f}%")
#         print(f"FCNN: Mean Accuracy: {np.mean(fcnn_accuracies) * 100:.2f}%, SD: {np.std(fcnn_accuracies) * 100:.2f}%")
#         print(
#             f"Transformer: Mean Accuracy: {np.mean(transformer_accuracies) * 100:.2f}%, SD: {np.std(transformer_accuracies) * 100:.2f}%")
#
#         return {
#             'lstm': {'accuracy_mean': np.mean(lstm_accuracies), 'accuracy_std': np.std(lstm_accuracies)},
#             'gru': {'accuracy_mean': np.mean(gru_accuracies), 'accuracy_std': np.std(gru_accuracies)},
#             'fcnn': {'accuracy_mean': np.mean(fcnn_accuracies), 'accuracy_std': np.std(fcnn_accuracies)},
#             'transformer': {'accuracy_mean': np.mean(transformer_accuracies),
#                             'accuracy_std': np.std(transformer_accuracies)}
#         }
#
# def main():
#     # input_dir = "/mnt/data/shyam/anushka/testing/created/trial2/data_files_sure/"
#     # asd_genes = "/mnt/data/shyam/anushka/testing/created/trial2/sfari_gene_selected.csv"
#     # asd_data_file = "/mnt/data/shyam/anushka/testing/created/trial2/asd_labels.csv"
#     # output_csv = "/mnt/data/shyam/anushka/testing/created/trial2/processed_data.csv"
#     input_dir = "/home/ibab/sem4/created/annotated_GRCh38.99/"
#     asd_genes = "/home/ibab/sem4/created/sfari_gene_selected.csv"
#     asd_data_file = "/home/ibab/sem4/created/simra_python/asd_labels.csv"
#     output_csv = "/home/ibab/sem4"
#     vcf2df = VCFprocessor(input_dir, asd_genes, asd_data_file, output_csv)
#     df = vcf2df.vcf2csv()
#
#     classify = ClassificationModels(df)
#     ml_results = classify.ml_models()
#     dl_results = classify.dl_models()
#     print("\nMachine Learning Models Results:")
#     for model, metrics in ml_results.items():
#         print(f"{model}: Accuracy: {metrics['accuracy_mean'] * 100:.2f}% ± {metrics['accuracy_std'] * 100:.2f}%")
#
#     print("\nDeep Learning Models Results:")
#     for model, metrics in dl_results.items():
#         print(f"{model}: Accuracy: {metrics['accuracy_mean'] * 100:.2f}% ± {metrics['accuracy_std'] * 100:.2f}%")
#
# if __name__ == "__main__":
#     main()

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
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Input, LayerNormalization, MultiHeadAttention, Flatten, Reshape
import gc
import re
from scipy import sparse


class CSVDataset(Dataset):
    def __init__(self, noisy_data, original_data):
        self.noisy_tensor = torch.tensor(noisy_data.values, dtype=torch.float32)
        self.original_tensor = torch.tensor(original_data.values, dtype=torch.float32)
        del noisy_data, original_data

    def __len__(self):
        return len(self.noisy_tensor)

    def __getitem__(self, idx):
        return self.noisy_tensor[idx], self.original_tensor[idx]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, input_dim),
                                     nn.Sigmoid())

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class VCFprocessor:
    def __init__(self, input_dir, asd_genes, asd_data_file, output_csv):
        self.input_dir = input_dir
        self.asd_genes = set(pd.read_csv(asd_genes, header=None).iloc[:, 0].tolist())
        self.asd_data = pd.read_csv(asd_data_file, sep=',', dtype={'subject_sp_id': str, 'asd': 'category'})
        self.output_csv = output_csv

        self.gene_pattern = re.compile(r"ANN=.*?\|.*?\|.*?\|(.*?)\|")
        self.variant_pattern = re.compile(r"ANN=.*?\|(.*?)\|")
        self.impact_pattern = re.compile(r"ANN=.*?\|.*?\|(.*?)\|")

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
        return format_field[:format_field.find(":")]

    def process_vcf_file(self, filename):
        """Process a single VCF file - extracted to reduce code duplication"""
        print(f"Processing: {filename}")
        lines_skipped = 0
        gvcf_file = os.path.join(self.input_dir, filename)
        file_name = os.path.splitext(filename)[0]

        data = []
        # Use context manager with buffered reading
        with open(gvcf_file, "r", buffering=1024 * 1024) as f:  # 1MB buffer
            for line in f:
                if line.startswith("#") or "\x00" in line:
                    if "\x00" in line:
                        lines_skipped += 1
                    continue

                columns = line.strip().split("\t")
                if len(columns) < 10:  # Ensure we have enough columns
                    continue

                ref = columns[3]  # REF column
                alt = columns[4]  # ALT column
                info = columns[7]  # INFO field
                format_field = columns[9]  # FORMAT field for zygosity

                # Extract required fields
                gene_name = self.extract_gene_name(info)
                if not any(gene in self.asd_genes for gene in gene_name.split(",")):
                    continue

                variant_type = self.extract_variant_type(info)
                impact = self.extract_impact(info)
                zygosity = self.extract_zygosity(format_field)

                data.append({"gene_name": gene_name, "ref": ref, "alt": alt, "variant_type": variant_type, "impact": impact, "zygosity": zygosity})

        if lines_skipped > 0:
            print(f"Lines skipped in {filename}: {lines_skipped} (due to null values being present)")

        if not data:
            print(f"No data found in {filename}, skipping")
            return None

        # Create DataFrame only after filtering
        df = pd.DataFrame(data)

        if df.empty or df.memory_usage(deep=True).sum() / 1024 <= 10:
            print(f"Insufficient data in {filename}, skipping")
            return None


        columns_to_encode = ["ref", "alt", "impact", "zygosity"]
        one_hot_encoded_df = pd.DataFrame()
        one_hot_encoded_df["gene_name"] = df["gene_name"]
        encoded_data = []
        for col in columns_to_encode:
            # Convert to categorical first for memory efficiency
            df[col] = pd.Categorical(df[col])
            one_hot = pd.get_dummies(df[col], prefix=col, dtype=np.int8)  # Use int8 instead of int64
            one_hot_encoded_df = pd.concat([one_hot_encoded_df, one_hot], axis=1)
        del df

        noisy_df = one_hot_encoded_df.copy()
        numeric_cols = [col for col in noisy_df.columns if col != "gene_name" and
                        pd.api.types.is_numeric_dtype(noisy_df[col])]
        noise_mask = (np.random.rand(len(noisy_df), len(numeric_cols)) < 0.1)
        for i, col in enumerate(numeric_cols):
            # Only modify rows where value is 1 and noise mask is True
            noisy_df.loc[(noisy_df[col] == 1) & noise_mask[:, i], col] = 0

        # Remove the gene_name column for the dataset
        noisy_data = noisy_df.drop("gene_name", axis=1)
        original_data = one_hot_encoded_df.drop("gene_name", axis=1)

        return self.train_autoencoder(noisy_data, original_data, file_name)

    def train_autoencoder(self, noisy_data, original_data, file_name):
        """Train autoencoder and extract features - extracted to separate method"""
        # Use pin_memory for faster GPU transfer if available
        use_cuda = torch.cuda.is_available()
        pin_memory = use_cuda

        dataset = CSVDataset(noisy_data, original_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                                pin_memory=pin_memory, num_workers=4)
        del noisy_data, original_data

        # Initialize Autoencoder
        input_dim = dataset.noisy_tensor.shape[1]
        model = Autoencoder(input_dim=input_dim, latent_dim=32)

        if use_cuda:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Train the model
        model.train()
        for epoch in range(10):  # 10 epochs as in original
            total_loss = 0
            for noisy_batch, original_batch in dataloader:
                if use_cuda:
                    noisy_batch = noisy_batch.cuda()
                    original_batch = original_batch.cuda()

                optimizer.zero_grad()
                reconstructed, _ = model(noisy_batch)
                loss = criterion(reconstructed, original_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

        # Extract latent representation
        model.eval()
        with torch.no_grad():
            # Process in smaller batches to reduce memory usage
            all_tensors = []
            batch_size = 128

            for i in range(0, len(dataset), batch_size):
                batch_end = min(i + batch_size, len(dataset))
                batch_tensor = dataset.noisy_tensor[i:batch_end]

                if use_cuda:
                    batch_tensor = batch_tensor.cuda()

                latent_batch = model.encoder(batch_tensor)
                all_tensors.append(latent_batch.cpu())  # Move back to CPU

            latent_vector = torch.cat(all_tensors).mean(dim=0, keepdim=True)

        # Free up GPU memory
        if use_cuda:
            torch.cuda.empty_cache()

        # Save latent representations to a DataFrame
        latent_df = pd.DataFrame(latent_vector.numpy())
        latent_df.columns = [f"latent_dim_{i + 1}" for i in range(latent_df.shape[1])]
        latent_df.insert(0, "file_name", file_name)

        return latent_df

    def vcf2csv(self):
        aggregated_df = pd.DataFrame()

        # Process files in batches to control memory usage
        vcf_files = [f for f in os.listdir(self.input_dir) if f.endswith(".gvcf")]

        # Process each file and add results to aggregated dataframe
        for filename in vcf_files:
            latent_df = self.process_vcf_file(filename)
            if latent_df is not None:
                aggregated_df = pd.concat([aggregated_df, latent_df], ignore_index=True)

            # Force garbage collection to free memory
            gc.collect()

        # Add target labels efficiently
        final_updated_data = aggregated_df.copy()
        final_updated_data["target"] = pd.NA  # Use pandas NA for initial values

        # Create a mapping dictionary for faster lookups
        asd_dict = dict(zip(self.asd_data["subject_sp_id"],
                            self.asd_data["asd"].map({"True": 1, "False": 0})))

        # Apply mapping function instead of iterating through rows
        final_updated_data["target"] = final_updated_data["file_name"].map(asd_dict)

        # Remove rows with no target
        final_updated_data = final_updated_data.dropna(subset=["target"])

        # Convert target to integer
        final_updated_data["target"] = final_updated_data["target"].astype(int)

        # Save to CSV
        final_updated_data.to_csv(self.output_csv, index=False)
        print(f"Processed final csv file: {self.output_csv}")

        return final_updated_data


class ClassificationModels:
    def __init__(self, df):
        self.df = df

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

        # Logistic Regression
        model = LogisticRegression(solver='saga', n_jobs=-1)  # Saga is memory efficient for large datasets
        results['logistic_regression'] = self.calculate_metrics(model, x, y, kf)
        del model

        # Random Forest - use reduced n_estimators and max_depth for memory efficiency
        model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        results['random_forest'] = self.calculate_metrics(model, x, y, kf)
        del model

        # XGBoost with memory efficient settings
        model = XGBClassifier(tree_method='hist', gpu_hist=True if torch.cuda.is_available() else False)
        results['xgboost'] = self.calculate_metrics(model, x, y, kf)
        del model

        # SVM - Use linear kernel which is more memory efficient
        model = SVC(kernel="linear", C=1.0, probability=True)
        results['svm'] = self.calculate_metrics(model, x, y, kf)
        del model

        return results

    def dl_models(self):
        # Setting up Tensorflow to use memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except Exception as e:
                print(f"Error setting memory growth: {e}")

        def build_lstm_model(input_dim, lstm_units=32):  # Reduced units for memory efficiency
            model = Sequential()
            model.add(LSTM(lstm_units, activation="relu", input_shape=(input_dim, 1)))
            model.add(Dropout(0.3))
            model.add(Dense(16, activation="relu"))  # Reduced units
            model.add(Dense(1, activation="sigmoid"))

            # Use Adam with lower learning rate for stability
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
            return model

        def build_gru_model(input_dim, rnn_units=32):  # Reduced units
            model = Sequential()
            model.add(GRU(rnn_units, activation="relu", input_shape=(input_dim, 1)))
            model.add(Dropout(0.3))
            model.add(Dense(16, activation="relu"))  # Reduced units
            model.add(Dense(1, activation="sigmoid"))

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
            return model

        def build_fcnn_model(input_dim, dense_units=32):  # Reduced units
            model = Sequential()
            model.add(Dense(dense_units, activation="relu", input_shape=(input_dim,)))
            model.add(Dropout(0.3))
            model.add(Dense(16, activation="relu"))  # Reduced units
            model.add(Dense(1, activation="sigmoid"))

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
            return model

        def build_transformer_model(input_dim):
            inputs = Input(shape=(input_dim,))
            x = Reshape((input_dim, 1))(inputs)

            # Reduced transformer dimensions
            x = Dense(32, activation="relu")(x)  # Reduced from 64
            x = LayerNormalization()(x)
            x = Dropout(0.2)(x)
            attention_output = MultiHeadAttention(num_heads=2, key_dim=8)(x, x)  # Reduced heads and key_dim
            x = attention_output + x  # Skip connection
            x = LayerNormalization()(x)
            x = Dense(16, activation="relu")(x)  # Reduced from 32
            x = Dropout(0.2)(x)
            x = Flatten()(x)
            outputs = Dense(1, activation="sigmoid")(x)
            model = Model(inputs=inputs, outputs=outputs)

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
            return model

        # Use float32 to save memory
        data = self.df.copy()
        # Use a smaller sample if dataset is large
        if len(data) > 300:  # Reduced from 500
            data = data.iloc[:300, :]

        data.drop("file_name", axis=1, inplace=True)
        data.fillna(0, inplace=True)

        # Convert to smaller dtypes
        for col in data.columns:
            if col != "target" and data[col].dtype == np.float64:
                data[col] = data[col].astype(np.float32)

        x = data.drop("target", axis=1).values.astype(np.float32)
        y = data["target"].values

        # Use smaller batch sizes and fewer epochs
        batch_size = 16  # Reduced from 32
        epochs = 20  # Reduced from 30

        # Smaller number of folds to save memory and time
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        lstm_accuracies = []
        gru_accuracies = []
        fcnn_accuracies = []
        transformer_accuracies = []

        for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Reshape for RNN models
            x_train_rnn = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_test_rnn = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

            # Train models with early stopping to prevent overfitting and save memory
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )

            # LSTM Model
            lstm_model = build_lstm_model(input_dim=x_train.shape[1])
            history = lstm_model.fit(
                x_train_rnn, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stopping]
            )
            loss, accuracy = lstm_model.evaluate(x_test_rnn, y_test, verbose=0)
            print(f"Fold {fold}: LSTM Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            lstm_accuracies.append(accuracy)

            # Clear model to free memory
            tf.keras.backend.clear_session()
            del lstm_model

            # GRU Model
            gru_model = build_gru_model(input_dim=x_train.shape[1])
            history = gru_model.fit(
                x_train_rnn, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stopping]
            )
            loss, accuracy = gru_model.evaluate(x_test_rnn, y_test, verbose=0)
            print(f"Fold {fold}: GRU Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            gru_accuracies.append(accuracy)

            # Clear model to free memory
            tf.keras.backend.clear_session()
            del gru_model

            # FCNN Model
            fcnn_model = build_fcnn_model(input_dim=x_train.shape[1])
            history = fcnn_model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stopping]
            )
            loss, accuracy = fcnn_model.evaluate(x_test, y_test, verbose=0)
            print(f"Fold {fold}: FCNN Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            fcnn_accuracies.append(accuracy)

            # Clear model to free memory
            tf.keras.backend.clear_session()
            del fcnn_model

            # Transformer Model
            transformer_model = build_transformer_model(input_dim=x_train.shape[1])
            history = transformer_model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stopping]
            )
            loss, accuracy = transformer_model.evaluate(x_test, y_test, verbose=0)
            print(f"Fold {fold}: Transformer Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            transformer_accuracies.append(accuracy)

            # Clear model to free memory
            tf.keras.backend.clear_session()
            del transformer_model

        print(f"LSTM: Mean Accuracy: {np.mean(lstm_accuracies) * 100:.2f}%, SD: {np.std(lstm_accuracies) * 100:.2f}%")
        print(f"GRU: Mean Accuracy: {np.mean(gru_accuracies) * 100:.2f}%, SD: {np.std(gru_accuracies) * 100:.2f}%")
        print(f"FCNN: Mean Accuracy: {np.mean(fcnn_accuracies) * 100:.2f}%, SD: {np.std(fcnn_accuracies) * 100:.2f}%")
        print(
            f"Transformer: Mean Accuracy: {np.mean(transformer_accuracies) * 100:.2f}%, SD: {np.std(transformer_accuracies) * 100:.2f}%")

        return {
            'lstm': {'accuracy_mean': np.mean(lstm_accuracies), 'accuracy_std': np.std(lstm_accuracies)},
            'gru': {'accuracy_mean': np.mean(gru_accuracies), 'accuracy_std': np.std(gru_accuracies)},
            'fcnn': {'accuracy_mean': np.mean(fcnn_accuracies), 'accuracy_std': np.std(fcnn_accuracies)},
            'transformer': {'accuracy_mean': np.mean(transformer_accuracies),
                            'accuracy_std': np.std(transformer_accuracies)}
        }


def main():
    os.environ['TF_MEMORY_ALLOCATION'] = 'growth'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    input_dir = "/home/ibab/sem4/created/annotated_GRCh38.99/"
    asd_genes = "/home/ibab/sem4/created/sfari_gene_selected.csv"
    asd_data_file = "/home/ibab/sem4/created/simra_python/asd_labels.csv"
    output_csv = "/home/ibab/sem4/final_dataset.csv"

    vcf2df = VCFprocessor(input_dir, asd_genes, asd_data_file, output_csv)
    df = vcf2df.vcf2csv()

    classify = ClassificationModels(df)
    ml_results = classify.ml_models()
    dl_results = classify.dl_models()

    print("\nMachine Learning Models Results:")
    for model, metrics in ml_results.items():
        print(f"{model}: Accuracy: {metrics['accuracy_mean'] * 100:.2f}% ± {metrics['accuracy_std'] * 100:.2f}%")

    print("\nDeep Learning Models Results:")
    for model, metrics in dl_results.items():
        print(f"{model}: Accuracy: {metrics['accuracy_mean'] * 100:.2f}% ± {metrics['accuracy_std'] * 100:.2f}%")


if __name__ == "__main__":
    main()
