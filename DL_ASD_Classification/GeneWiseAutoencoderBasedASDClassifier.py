import csv
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, SimpleRNN, Input, LayerNormalization, MultiHeadAttention, Flatten, Softmax, Reshape
from tensorflow.keras.optimizers import Adam

def extract_gene_name(info):
    # Extract gene name(s) from INFO field (ANN annotation)
    if "ANN=" in info:
        annotations = info.split("ANN=")[1].split(",")
        gene_names = set()
        for annotation in annotations:
            parts = annotation.split("|")
            if len(parts) > 3:  # Check if gene name exists in the annotation
                gene_names.add(parts[3])
        return ",".join(gene_names)
    return "Unknown"

def extract_variant_type(info):
    # Extract variant type from INFO field (ANN annotation)
    if "ANN=" in info:
        annotations = info.split("ANN=")[1].split(",")
        variant_types = set()
        for annotation in annotations:
            parts = annotation.split("|")
            if len(parts) > 1:  # Check if variant type exists in the annotation
                variant_types.add(parts[1])
        return ",".join(variant_types)
    return "Unknown"

def extract_impact(info):
    # Extract impact from INFO field
    if "ANN=" in info:
        annotations = info.split("ANN=")[1].split(",")
        impacts = set()
        for annotation in annotations:
            parts = annotation.split("|")
            if len(parts) > 2:  # Check if impact exists in the annotation
                impacts.add(parts[2])
        return ",".join(impacts)
    return "Unknown"

def extract_zygosity(format_field):
    # Extract zygosity from FORMAT field
    return format_field.split(":")[0]

def vcf2csv():
    input_dir = "/home/ibab/sem4/datasets/data_processed"
    output_dir = "/home/ibab/sem4/created/anushka_python/trial1/vcf2csv"

    for filename in os.listdir(input_dir):
        if filename.endswith(".gvcf"):
            print(filename)
            gvcf_file = os.path.join(input_dir, filename)
            csv_file = os.path.join(output_dir, filename.replace(".gvcf", ".csv"))

            with open(gvcf_file, "r") as f:
                reader = f.readlines()

            with open(csv_file, "w", newline='') as csvfile:
                fieldnames = ["gene_name", "ref", "alt", "variant_type", "impact", "zygosity"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for line in reader:
                    if line.startswith("#"):
                        continue
                    columns = line.strip().split("\t")
                    ref = columns[3]  # REF column
                    alt = columns[4]  # ALT column
                    info = columns[7]  # INFO field
                    format_field = columns[9]  # FORMAT field for zygosity

                    # Extract required fields
                    gene_name = extract_gene_name(info)
                    variant_type = extract_variant_type(info)
                    impact = extract_impact(info)
                    zygosity = extract_zygosity(format_field)

                    writer.writerow({"gene_name": gene_name, "ref": ref, "alt": alt, "variant_type": variant_type, "impact": impact, "zygosity": zygosity})

def only_asd_genes():
    input_dir = "/home/ibab/sem4/created/anushka_python/trial1/vcf2csv/"
    asd_genes = "/home/ibab/sem4/created/sfari_gene_selected.csv"
    output_dir = "/home/ibab/sem4/created/anushka_python/trial1/only_asd_genes"

    # Extract teh ASD relevant genes from teh csv file
    df = pd.read_csv(asd_genes, header=None)
    gene_list = df.iloc[:, 0].tolist()

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path)

            def gene_matches(row):
                genes = str(row).split(",")
                return any(gene in gene_list for gene in genes)

            filtered_df = df[df["gene_name"].apply(gene_matches)]
            output_path = os.path.join(output_dir, filename)
            filtered_df.to_csv(output_path, index=False)

def one_hot_csv():
    # This code will one hot encode all the columns that are specified in columns_to_encode + retain gene_names + delete the rest
    input_dir = "/home/ibab/sem4/created/anushka_python/trial1/only_asd_genes/"
    output_dir = "/home/ibab/sem4/created/anushka_python/trial1/one_hot_csv/"

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            input_path = os.path.join(input_dir, file_name)

            if os.path.getsize(input_path) > 10:
                df = pd.read_csv(input_path)
                columns_to_encode = ["ref", "alt", "impact", "zygosity"]
                one_hot_encoded_df = pd.DataFrame()
                one_hot_encoded_df["gene_name"] = df["gene_name"]

                for col in columns_to_encode:
                    one_hot = pd.get_dummies(df[col], prefix=col, dtype=int)
                    one_hot_encoded_df = pd.concat([one_hot_encoded_df, one_hot], axis=1)

                output_path = os.path.join(output_dir, file_name)
                one_hot_encoded_df.to_csv(output_path, index=False)
            else:
                print("Skipping empty file: ", input_path)

def noisy_one_hot_csv():
    # This code will add masking noise to one hot encoded .csv files
    # Masking Noise: Random 1s are flipped to 0s
    input_dir = "/home/ibab/sem4/created/anushka_python/trial1/one_hot_csv/"
    output_dir = "/home/ibab/sem4/created/anushka_python/trial1/noisy_one_hot_csv/"

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            df = pd.read_csv(input_path)
            noisy_df = df.copy()

            for col in noisy_df.columns:
                if noisy_df[col].dtype in [np.int64, np.float64]:  # Ensure it's numeric (0/1)
                    mask = (noisy_df[col] == 1) & (np.random.rand(len(noisy_df[col])) < 0.1)  # 10% probability
                    noisy_df.loc[mask, col] = 0  # Flip 1s to 0s

            noisy_df.to_csv(output_path, index=False)

#v1:  class CSVGeneDataset(Dataset):
#     def __init__(self, csv_file, gene_name, one_hot_dir):
#         self.data = pd.read_csv(csv_file)
#         self.gene_name = gene_name
#         self.data = self.data[self.data["gene_name"] == gene_name].drop(columns=["gene_name"])
#         self.data_tensor = torch.tensor(self.data.values, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.data_tensor)
#
#     def __getitem__(self, idx):
#         return self.data_tensor[idx]

class CSVGeneDataset(Dataset):
    # def __init__(self, noisy_csv_file, original_csv_file):
    #     self.noisy_data = pd.read_csv(noisy_csv_file).iloc[:, 1:]
    #     self.original_data = pd.read_csv(original_csv_file).iloc[:, 1:]
    #     self.noisy_tensor = torch.tensor(self.noisy_data.values, dtype=torch.float32)
    #     self.original_tensor = torch.tensor(self.original_data.values, dtype=torch.float32)

    def __init__(self, noisy_data, original_data):
        self.noisy_data = noisy_data
        self.original_data = original_data
        self.noisy_tensor = torch.tensor(self.noisy_data.values, dtype=torch.float32)
        self.original_tensor = torch.tensor(self.original_data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.noisy_tensor)

    def __getitem__(self, idx):
        return self.noisy_tensor[idx], self.original_tensor[idx]

class Autoencoder(nn.Module):
    # def __init__(self, input_dim, latent_dim):
    #     super(Autoencoder, self).__init__()
    #     self.encoder = nn.Sequential(nn.Linear(input_dim, 128),
    #                                  nn.ReLU(),
    #                                  nn.Linear(128, latent_dim)
    #                                  )
    #     self.decoder = nn.Sequential(nn.Linear(latent_dim, 128),
    #                                  nn.ReLU(),
    #                                  nn.Linear(128, input_dim),
    #                                  nn.Sigmoid()
    #                                  )

    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, latent_dim)
                                     )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, input_dim),
                                     nn.Sigmoid()
                                     )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

#v1: def train_autoencoder(model, dataloader, optimizer, criterion, epochs=10):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             optimizer.zero_grad()
#             reconstructed, _ = model(batch)
#             loss = criterion(reconstructed, batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

def train_autoencoder(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for noisy_batch, original_batch in dataloader:
            optimizer.zero_grad()
            reconstructed, _ = model(noisy_batch)
            loss = criterion(reconstructed, original_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

def generate_context_autoencoder():
    noisy_dir = "/home/ibab/sem4/created/anushka_python/trial1/noisy_one_hot_csv/"
    original_dir = "/home/ibab/sem4/created/anushka_python/trial1/one_hot_csv/"
    output_dir = "/home/ibab/sem4/created/anushka_python/trial1/context/"

    # Iterate through noisy_dir
    for file_name in os.listdir(noisy_dir):
        if file_name.endswith(".csv"):
            noisy_file = os.path.join(noisy_dir, file_name)
            original_file = os.path.join(original_dir, file_name)

            # Load the original file to get unique gene names
            noisy_data = pd.read_csv(noisy_file)
            original_data = pd.read_csv(original_file)
            unique_genes = original_data["gene_name"].unique()

            # Initialize a list to store context vectors for this file
            context_data = []

            # Process each unique gene
            for u_gene_name in unique_genes:
                u_gene_noisy_data = noisy_data[noisy_data["gene_name"].isin([u_gene_name])].iloc[:, 1:]
                u_gene_original_data = original_data[original_data["gene_name"].isin([u_gene_name])].iloc[:, 1:]
                dataset = CSVGeneDataset(u_gene_noisy_data, u_gene_original_data)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                # Initialize Autoencoder model
                #v1: input_dim = dataset.data.shape[1]
                input_dim = dataset.noisy_tensor.shape[1]
                model = Autoencoder(input_dim=input_dim, latent_dim=128)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()

                # Train the model
                train_autoencoder(model, dataloader, optimizer, criterion, epochs=10)

                # Extract the context vector for the gene
                model.eval()
                with torch.no_grad():
                    #v1: context_vector = model.encoder(dataset.data_tensor).mean(dim=0).numpy()
                    context_vector = model.encoder(dataset.noisy_tensor).mean(dim=0, keepdim=True)

                # Store the gene name and context vector
                context_data.append([u_gene_name] + context_vector.tolist())

            # Convert the context data to a DataFrame
            context_df = pd.DataFrame(context_data)
            context_df.columns = ["gene_name"] + [f"context_vector_{i}" for i in range(1, len(context_df.columns))]

            # Save the context DataFrame to a new CSV file
            output_file = os.path.join(output_dir, f"{file_name}")
            context_df.to_csv(output_file, index=False)
            print("Processed: ", file_name)

def context_proper_csv():
    input_dir = "/home/ibab/sem4/created/anushka_python/trial1/context/"
    output_dir = "/home/ibab/sem4/created/anushka_python/trial1/context_proper/"

    for file_name in os.listdir(input_dir):
        print(file_name)
        input_file = os.path.join(input_dir, file_name)
        df = pd.read_csv(input_file)
        vector_cols = df["context_vector_1"].str.strip("[]").str.split(",", expand=True)
        vector_cols.columns = [f"vector_{i + 1}" for i in range(vector_cols.shape[1])] # Rename the columns
        df = pd.concat([df[["gene_name"]], vector_cols], axis=1)
        output_file = os.path.join(output_dir, f"{file_name}")
        df.to_csv(output_file, index=False)
        print("Processed: ", file_name)

def calculate_mean_context():
    context_dir = "/home/ibab/sem4/created/anushka_python/trial1/context_proper/"
    mean_context_dir = "/home/ibab/sem4/created/anushka_python/trial1/mean_context/"

    for file_name in os.listdir(context_dir):
        if file_name.endswith(".csv"):
            context_file = os.path.join(context_dir, file_name)
            df = pd.read_csv(context_file)

            df["mean_context"] = df.iloc[:, 1:].mean(axis=1)
            # df["mean_context"] = df.iloc[:, 1:].apply(lambda row: ((row - row.mean()).sum()) / (row.std()/np.sqrt(32)), axis=1)
            mean_context_df = df[["gene_name", "mean_context"]]

            output_file = os.path.join(mean_context_dir, file_name)
            mean_context_df.to_csv(output_file, index=False)
    print("Mean context for all files calculated")

def generate_dataset():
    mean_context_dir = "/home/ibab/sem4/created/anushka_python/trial1/mean_context/"
    output_dir = "/home/ibab/sem4/created/anushka_python/trial1/"
    output_file = os.path.join(output_dir, "aggregated_dataset.csv")

    # Collect all unique gene names from all CSV files
    all_gene_names = set()
    csv_data = {}

    for file_name in os.listdir(mean_context_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(mean_context_dir, file_name)
            df = pd.read_csv(file_path)

            # Add the gene names to the set
            all_gene_names.update(df["gene_name"].unique())

            # Save the DataFrame in a dictionary with the file name (without .csv) as the key
            csv_data[file_name.replace(".csv", "")] = df

    # Convert the set of gene names to a sorted list
    all_gene_names = sorted(all_gene_names)

    # Initialize the output DataFrame
    aggregated_df = pd.DataFrame(columns=["file_name"] + all_gene_names)

    # Populate the DataFrame
    for file_name, df in csv_data.items():
        # Create a dictionary for this file with gene_name as keys and mean_context as values
        mean_context_map = dict(zip(df["gene_name"], df["mean_context"]))

        # Create a row with file_name and mean_context values (or 0 if gene_name is missing)
        row = [file_name] + [mean_context_map.get(gene_name, 0) for gene_name in all_gene_names]

        # Append the row to the DataFrame
        aggregated_df.loc[len(aggregated_df)] = row

    # Save the resulting DataFrame to a single CSV file
    aggregated_df.to_csv(output_file, index=False)

def generate_dataset_target():
    # Read the input files
    final_updated_data = pd.read_csv("/home/ibab/sem4/created/anushka_python/trial1/aggregated_dataset.csv")
    asd_data = pd.read_csv("/home/ibab/sem4/created/simra_python/asd_labels.csv", sep=',', dtype={'asd': str})
    output = "/home/ibab/sem4/created/anushka_python/trial1/aggregated_data_target.csv"
    final_updated_data["target"] = ""
    rows_to_drop = []

    for index, row in final_updated_data.iterrows():
        sample_name = row["file_name"]

        # Find the matching row in asd_data
        matching_row = asd_data.loc[asd_data["subject_sp_id"] == sample_name]

        if not matching_row.empty:
            asd_value = matching_row.iloc[0]["asd"]
            if asd_value == "True":
                final_updated_data.loc[index, 'target'] = 1
            elif asd_value == 'False':
                final_updated_data.loc[index, 'target'] = 0
        else:
            rows_to_drop.append(index)
            print(f"No matching sample name found in ASD data for {sample_name}")

    final_updated_data.drop(index=rows_to_drop, inplace=True)
    final_updated_data.to_csv(output, index=False)
    print("Updated data has been saved with the 'target' column.")

def calculate_metrics(model, x, y, kf):
    accuracy_scores = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    print(f" {model}, Mean: {np.mean(accuracy_scores) * 100}, SD: {np.std(accuracy_scores) * 100}")

def ml_models():
    data = pd.read_csv("/home/ibab/sem4/created/anushka_python/trial1/aggregated_data_target.csv")
    data.drop("file_name", axis=1, inplace=True)
    data.fillna(0, inplace=True)
    x = data.drop("target", axis=1)
    y = data["target"]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    model = LogisticRegression()
    calculate_metrics(model, x, y, kf)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    calculate_metrics(model, x, y, kf)
    model = XGBClassifier()
    calculate_metrics(model, x, y, kf)
    model = SVC(kernel="linear", C=1.0, probability=True)
    calculate_metrics(model, x, y, kf)

def build_lstm_model(input_dim, lstm_units=64, dense_units=32, dropout_rate=0.3):
    model = Sequential()
    model.add(LSTM(lstm_units, activation="relu", input_shape=(input_dim, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_gru_model(input_dim, rnn_units=64, dense_units=32, dropout_rate=0.3):
    model = Sequential()
    model.add(GRU(rnn_units, activation="relu", input_shape=(input_dim, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_fcnn_model(input_dim, dense_units=64, dropout_rate=0.3):
    model = Sequential()
    model.add(Dense(dense_units, activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units // 2, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_transformer_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    # Transformer block
    x = Dense(64, activation="relu")(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Reshape((-1,))(x)
    outputs = Dense(2, activation="softmax")(x)  # Number of classes = 2
    model = Model(inputs, outputs)
    return model

def dl_models():
    data = pd.read_csv("/home/ibab/sem4/created/anushka_python/trial1/aggregated_data_target.csv")
    data = data.iloc[200:300, :]
    data.drop("file_name", axis=1, inplace=True)
    data.fillna(0, inplace=True)
    x = data.drop("target", axis=1).values
    y = data["target"].values
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    lstm_accuracies = []
    gru_accuracies = []
    fcnn_accuracies = []
    transformer_accuracies = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train_rnn = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test_rnn = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        lstm_model = build_lstm_model(input_dim=x_train_rnn.shape[1])
        lstm_model.fit(x_train_rnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        y_pred = (lstm_model.predict(x_test_rnn) > 0.5).astype(int).flatten()
        lstm_accuracies.append(accuracy_score(y_test, y_pred))

        gru_model = build_gru_model(input_dim=x_train_rnn.shape[1])
        gru_model.fit(x_train_rnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        y_pred = (gru_model.predict(x_test_rnn) > 0.5).astype(int).flatten()
        gru_accuracies.append(accuracy_score(y_test, y_pred))

        fcnn_model = build_fcnn_model(input_dim=x_train.shape[1])
        fcnn_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        y_pred = (fcnn_model.predict(x_test) > 0.5).astype(int).flatten()
        fcnn_accuracies.append(accuracy_score(y_test, y_pred))

        transformer_model = build_transformer_model(input_dim=x_train_rnn.shape[1])
        # transformer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        transformer_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        transformer_model.fit(x_train_rnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        # y_pred = (transformer_model.predict(x_test_rnn) > 0.5).astype(int).flatten()
        # transformer_accuracies.append(accuracy_score(y_test, y_pred))
        y_pred = transformer_model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
        accuracy = accuracy_score(y_test, y_pred_classes)
        transformer_accuracies.append(accuracy)
    print(f"LSTM: {np.mean(lstm_accuracies) * 100}, SD: {np.std(lstm_accuracies) * 100}")
    print(f"GRU: {np.mean(gru_accuracies) * 100}, SD: {np.std(gru_accuracies) * 100}")
    print(f"FCNN:{np.mean(fcnn_accuracies) * 100}, SD: {np.std(fcnn_accuracies) * 100}")
    print(f"Transformer: {np.mean(transformer_accuracies) * 100}, SD: {np.std(transformer_accuracies) * 100}")

def main():
    vcf2csv() # Input: Annotated snpEff files, Extract relevant details from vcf files and convert to csv
    only_asd_genes() # Filter out genes that are not involved in ASD
    one_hot_csv() # One hot encode the features of the csv files
    noisy_one_hot_csv() # Add masking noise
    generate_context_autoencoder() # Train AutoEncoder to learn the latent representation of each gene in each csv file
    context_proper_csv() # Correctly format the context vector values to a csv file
    calculate_mean_context() # Boil down the latent vector (1x32 dim) to a 1x1 dim number by taking mean
    generate_dataset() # Combine the values from all csv files to come to a single csv file
    generate_dataset_target() # Add target values (ASD/Not)
    ml_models() # Run ML models on the dataset
    dl_models() # Run DL models on the dataset

if __name__ == "__main__":
    main()
