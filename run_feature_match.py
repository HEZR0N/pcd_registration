import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Define the transformation prediction model
class FeatureMatchingModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=7):
        super(FeatureMatchingModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)


# Custom dataset class for the point clouds and transformations
class PointCloudDataset(Dataset):
    def __init__(self, sources, targets, transformations):
        self.sources = sources
        self.targets = targets
        self.transformations = transformations

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        # target_pcd = target_pcd[~np.isnan(target_pcd).all(axis=1)]
        # source_pcd = source_pcd[~np.isnan(source_pcd).all(axis=1)]
        src = torch.tensor(self.sources[idx], dtype=torch.float32)
        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)
        trans = torch.tensor(self.transformations[idx], dtype=torch.float32)
        # Replace NaN values with 0
        src = torch.where(torch.isnan(src), torch.tensor(0.0, dtype=torch.float32), src)
        tgt = torch.where(torch.isnan(tgt), torch.tensor(0.0, dtype=torch.float32), tgt)
        trans = torch.where(torch.isnan(trans), torch.tensor(0.0, dtype=torch.float32), trans)
        return src, tgt, trans

# Training function
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=50):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for src, tgt, trans in dataloader:
            # Move data to the GPU (or CPU if GPU is unavailable) and convert to Float
            src = src.to(device).float()
            tgt = tgt.to(device).float()
            trans = trans.to(device).float()

            # Flatten the input data
            inputs = torch.cat([src.view(src.size(0), -1), tgt.view(tgt.size(0), -1)], dim=1)

            # Forward pass
            outputs = model(inputs) # Error: always returns [nan, nan, nan, nan, nan, nan, nan]
            # print(outputs)
            loss = criterion(outputs, trans)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(dataloader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return train_losses

# Main function to process data and train models
def process_and_train(data_path, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Load data
    with h5py.File(data_path, 'r') as file:
        noise_levels = [0, 1, 5, 10]
        lower_bounds = [55, 45, 35, 25]

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i, lower_bound in enumerate(lower_bounds):
            for j, noise_level in enumerate(noise_levels):
                ax_idx = i * 4 + j
                ax = axes[ax_idx]

                print(f"Training model for noise level {noise_level}%, lower bound {lower_bound}%...")
                src = file[f'{lower_bound}/noise_{noise_level}/src'][:]
                tgt = file[f'{lower_bound}/noise_{noise_level}/tgt'][:]
                transformations = file[f'{lower_bound}/noise_{noise_level}/rotations'][:]

                # Preprocess data
                src_train, src_val, tgt_train, tgt_val, trans_train, trans_val = train_test_split(
                    src, tgt, transformations, test_size=0.2, random_state=42
                )

                # Create datasets and dataloaders
                train_dataset = PointCloudDataset(src_train, tgt_train, trans_train)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                # Determine input size
                # print(train_dataset[0])
                sample_src, sample_tgt, _ = train_dataset[0]
                input_size = sample_src.numel() + sample_tgt.numel()
                print(input_size)

                # Initialize the model, loss function, and optimizer
                model = FeatureMatchingModel(input_size=input_size).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # Train the model
                train_losses = train_model(model, train_loader, criterion, optimizer, device, 20)

                # Save the trained model
                model_save_path = os.path.join(
                    output_dir, f"model_noise_{noise_level}_lb_{lower_bound}.pth"
                )
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}\n")

                # Plot the training loss
                ax.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", color="blue")
                ax.set_title(f"Noise: {noise_level}%, LB: {lower_bound}%")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.legend()

        plt.tight_layout()
        plot_save_path = os.path.join(output_dir, "training_plots.png")
        plt.savefig(plot_save_path, dpi=300)
        print(f"Training plots saved to {plot_save_path}")

if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths and parameters
    training_data_path = "train/train_data.h5"
    output_directory = "feature_match_models"

    # Process data and train models
    process_and_train(training_data_path, output_directory, device)
