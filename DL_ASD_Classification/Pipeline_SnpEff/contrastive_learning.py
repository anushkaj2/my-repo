import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Contrastive learning model for genomic data
# Args:
# input_dim: Dimension of input features
# projection_dim: Dimension of the projection head
# temperature: Temperature parameter for contrastive loss

class ContrastiveLearningModel(nn.Module):
    def __init__(self, input_dim, projection_dim=64, temperature=0.1):
        super(ContrastiveLearningModel, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(128, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.temperature = temperature

    def forward(self, x):
        # Forward pass through encoder and projection head
        #Args:
        # x: Input features
        # Returns: Projected features
        
        features = self.encoder(x)
        projections = self.projection_head(features)
        return projections, features

    def contrastive_loss(self, z_i, z_j):
        # Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
        # Normalize projections
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        batch_size = z_i.size(0)

        # Combine both views
        z = torch.cat([z_i, z_j], dim=0)  # [2*B, D]

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2*B, 2*B]

        # Remove self-similarity by masking
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

        # Compute loss
        labels = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size)
        ]).to(z.device)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


def data_augmentation(x, aug_type='noise', noise_factor=0.01):

    # Data augmentation for contrastive learning

    # Args:
    # - x: Input data
    # - aug_type: Type of augmentation ('noise', 'scale', 'flip')
    # - noise_factor: Strength of noise augmentation
    # Returns: Augmented data

    x = x.clone()

    if aug_type == 'noise':
        # Add Gaussian noise
        noise = torch.randn_like(x) * noise_factor
        return x + noise

    elif aug_type == 'scale':
        # Random scaling
        scale_factor = 1 + (torch.rand_like(x) - 0.5) * 0.2
        return x * scale_factor

    elif aug_type == 'flip':
        # Random sign flipping
        flip_mask = torch.rand_like(x) < 0.5
        x[flip_mask] *= -1
        return x

    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def train_contrastive_model(x, x_val, y_val, y_train, epochs=50, batch_size=64, learning_rate=1e-3):
    # Train contrastive learning model with validation metrics tracking
    # Args:
    # - x: Input features for training
    # - x_val: Input features for validation
    # - y_val: Labels for validation
    # - y_train: Labels for training
    # - epochs: Number of training epochs
    # - batch_size: Batch size for training
    # - learning_rate: Learning rate for optimizer
    # Returns:
    # - Trained contrastive model
    # - Encoded features
    # - History dictionary with loss, accuracy, and validation error metrics

    # Ensure input is torch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)
    
    if not isinstance(x_val, torch.Tensor):
        x_val = torch.FloatTensor(x_val)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Normalize data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_tensor = torch.FloatTensor(x_scaled).to(device)
    
    x_val_scaled = scaler.transform(x_val)
    x_val_tensor = torch.FloatTensor(x_val_scaled).to(device)

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(x_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss
    model = ContrastiveLearningModel(input_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize history dictionary to track metrics
    history = {
        'loss': [],
        'val_loss': [],  # Added validation loss tracking
        'val_error': [],
        'accuracy': []
    }

    # For validation classifier
    val_classifier = LogisticRegression(max_iter=1000)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_losses = []

        for batch in dataloader:
            # Get batch and create two augmented views
            x_batch = batch[0]

            # Create two different augmentations of the same batch
            x_aug1 = data_augmentation(x_batch, aug_type='noise')
            x_aug2 = data_augmentation(x_batch, aug_type='scale')

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            z1, _ = model(x_aug1)
            z2, _ = model(x_aug2)

            # Compute contrastive loss
            loss = model.contrastive_loss(z1, z2)

            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())

        # Calculate average loss for the epoch
        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)
        
        # Compute validation metrics
        model.eval()
        with torch.no_grad():
            # Get encoded features for training and validation
            _, train_features = model(x_tensor)
            _, val_features = model(x_val_tensor)
            
            # Compute validation loss (using same contrastive approach)
            x_val_aug1 = data_augmentation(x_val_tensor, aug_type='noise')
            x_val_aug2 = data_augmentation(x_val_tensor, aug_type='scale')
            val_z1, _ = model(x_val_aug1)
            val_z2, _ = model(x_val_aug2)
            val_loss = model.contrastive_loss(val_z1, val_z2).item()
            history['val_loss'].append(val_loss)
            
            # Convert to numpy for scikit-learn
            train_features_np = train_features.cpu().numpy()
            val_features_np = val_features.cpu().numpy()
            
            # Fit classifier on training features
            val_classifier.fit(train_features_np, y_train)
            
            # Predict on validation set
            val_preds = val_classifier.predict(val_features_np)
            
            # Calculate validation error and accuracy
            val_accuracy = accuracy_score(y_val, val_preds)
            val_error = 1 - val_accuracy
            
            history['val_error'].append(val_error)
            history['accuracy'].append(val_accuracy)
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Validation Accuracy: {val_accuracy:.4f}, "
                  f"Validation Error: {val_error:.4f}")
        
        # Set model back to training mode
        model.train()

    # Extract learned representations
    model.eval()
    with torch.no_grad():
        _, encoded_features = model(x_tensor)
        encoded_features = encoded_features.cpu().numpy()

    return model, encoded_features, history


def plot_transformer_style_plots(history, save_path):
    # Plot training and validation metrics in the TransformerModel style shown in the example
    
    # Args:
    # - history: Dictionary containing training metrics
    # - save_path: Path to save the plots to

    # Set style parameters
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 12)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.alpha'] = 0.7
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot Validation Accuracy
    ax1.plot(history['accuracy'], 'b-o', label='Validation Accuracy')
    ax1.set_title('ContrastiveLearningModel Validation Accuracy Over Epochs', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.set_ylim([0.55, 0.62])  # Similar to example
    
    # Plot Losses
    ax2.plot(history['loss'], 'r-o', label='Training Loss')
    ax2.plot(history['val_loss'], 'g-o', label='Validation Loss')
    ax2.set_title('ContrastiveLearningModel Losses Over Epochs', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()


def prepare_for_downstream_task(model, x, supervised=True, y=None):
    # Prepare features for downstream supervised or self-supervised task
    # Ensure input is torch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)

    # Normalize data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_tensor = torch.FloatTensor(x_scaled).to(next(model.parameters()).device)

    # Extract features using encoder
    model.eval()
    with torch.no_grad():
        _, encoded_features = model(x_tensor)
        encoded_features = encoded_features.cpu().numpy()

    if supervised and y is not None:
        return encoded_features, y

    return encoded_features


def calculate_sd_accuracy(model, x, y, n_splits=5):
    # Calculate standard deviation of accuracy across multiple folds
    
    # Args:
    # - model: Trained contrastive model
    # - x: Input features
    # - y: Labels
    # - n_splits: Number of cross-validation splits
    
    # Returns:
    # - Mean accuracy
    # - Standard deviation of accuracy
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    for train_idx, test_idx in kf.split(x):
        # Get data for this fold
        x_train_fold, x_test_fold = x[train_idx], x[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Get encoded features
        x_train_encoded = prepare_for_downstream_task(model, x_train_fold, supervised=False)
        x_test_encoded = prepare_for_downstream_task(model, x_test_fold, supervised=False)
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train_encoded, y_train_fold)
        
        # Evaluate
        y_pred = clf.predict(x_test_encoded)
        acc = accuracy_score(y_test_fold, y_pred)
        accuracies.append(acc)
    
    return np.mean(accuracies), np.std(accuracies)


def compare_contrastive_representations(x, y, test_size=0.2, random_state=42, epochs=50):
    # Compare representations learned through contrastive learning
    
    # Args:
    # - x: Input features
    # - y: Labels
    # - test_size: Proportion of test set
    # - random_state: Random seed for reproducibility
    # - epochs: Number of training epochs
    
    # Returns:
    # - Dictionary of performance metrics

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train contrastive model with validation
    contrastive_model, x_train_contrastive, history = train_contrastive_model(
        x_train, x_test, y_test, y_train, epochs=epochs
    )
    
    # Plot training history using the new transformer-style plot function
    plot_transformer_style_plots(
        history, "/mnt/data/shyam/anushka/testing/created/autos/transformer_style_plots.png"
    )
    
    # Calculate standard deviation of accuracy
    mean_acc, std_acc = calculate_sd_accuracy(contrastive_model, x, y)
    
    # Prepare test features
    x_test_contrastive = prepare_for_downstream_task(
        contrastive_model, x_test, supervised=False
    )
    
    # Train and evaluate linear classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train_contrastive, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(x_test_contrastive)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate detailed report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'contrastive_model': contrastive_model,
        'accuracy_mean': mean_acc,
        'accuracy_std': std_acc
    }


def main():
    import pandas as pd
    # df = pd.read_csv("/mnt/data/shyam/anushka/testing/created/autos/final_df.csv", on_bad_lines='skip')
    df = pd.read_csv("/mnt/data/shyam/anushka/testing/created/autos/vep_50kv2/latent_with_asd_labels.csv", on_bad_lines="skip")
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Prepare features and labels
    X = df.drop(['status', 'sample_id'], axis=1).values
    y = df['status'].values
    
    print("Starting contrastive learning training...")
    # Compare representations with fewer epochs (6 to match example plot)
    results = compare_contrastive_representations(X, y, epochs=6)
    
    # Print results
    print("\nContrastive Learning Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Mean Accuracy (across folds): {results['accuracy_mean']:.4f}")
    print(f"Standard Deviation of Accuracy: {results['accuracy_std']:.4f}")
    print("\nDetailed Classification Report:")
    from pprint import pprint
    pprint(results['classification_report'])
    
    print(f"\nTraining plot saved to: /mnt/data/shyam/anushka/testing/created/autos/transformer_style_plots.png")


if __name__ == "__main__":
    main()
