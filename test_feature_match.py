import os
import h5py
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader


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


# Function to calculate statistics
def calculate_statistics(diffs):
    mean = np.mean(diffs)
    stddev = np.std(diffs)
    median = np.median(diffs)
    return mean, stddev, median

# Function to calculate angle and translation differences
def calculate_differences(predicted_transformations, correct_transformations):
    angle_diffs = []
    translation_diffs = []
    # print(predicted_transformations)
    for pred_trans32, correct_trans32 in zip(predicted_transformations, correct_transformations):
        pred_trans32 = pred_trans32.cpu()
        pred_trans32 = pred_trans32.detach().numpy()
        correct_trans32 = correct_trans32.cpu()
        correct_trans32 = correct_trans32.detach().numpy()
        # print(pred_trans32.detach().numpy())
        # pred_rotation = R.from_matrix(pred_trans32[:3, :3])
        # pred_trans32lation = pred_trans32[:3, 3]
        # print(pred_trans32.shape)
        # print(correct_trans32.shape)
        for pred_trans, correct_trans in zip(pred_trans32, correct_trans32):
            # print(pred_trans.shape)
            # print(correct_trans.shape)
            pred_angle = pred_trans[:4]
            pred_translation = pred_trans[4:]
            pred_rotation = R.from_rotvec(pred_angle[:3] * np.radians(pred_angle[3]))
            pred_rotation_matrix = pred_rotation.as_matrix()

            cor_angle = correct_trans[:4]
            correct_translation = correct_trans[4:]
            correct_rotation = R.from_rotvec(cor_angle[:3] * np.radians(cor_angle[3]))
            correct_rotation_matrix = correct_rotation.as_matrix()
            

            # Angle difference
            # angle_diff = np.degrees(pred_rotation.inv() * correct_rotation).magnitude()
            # angle_diff = np.degrees(pred_rotation_matrix.inv() * correct_rotation).magnitude()
            angle_diff = np.degrees(np.arccos((np.trace(np.dot(correct_rotation_matrix.T, pred_rotation_matrix)) - 1) / 2))
            # Translation difference
            translation_diff = np.linalg.norm(pred_translation - correct_translation)

            angle_diffs.append(angle_diff)
            translation_diffs.append(translation_diff)
    return angle_diffs, translation_diffs

# Main testing and evaluation loop
def evaluate_models_old(models_dir, test_data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all models
    models = {f: torch.load(os.path.join(models_dir, f)) for f in os.listdir(models_dir) if f.endswith('.pth')}
    
    # Load test datasets
    with h5py.File(test_data_path, 'r') as test_file:
        datasets = {(noise, lb): (test_file[f"{lb}/noise_{noise}"]["src"][:], 
                                  test_file[f"{lb}/noise_{noise}"]["tgt"][:],
                                  test_file[f"{lb}/noise_{noise}"]["rotations"][:])
                    for lb in [25, 35, 45, 55]
                    for noise in [0, 1, 5, 10]}

    for model_name, model in models.items():
        model_output_dir = os.path.join(output_dir, model_name.replace(".pth", ""))
        os.makedirs(model_output_dir, exist_ok=True)

        losses, angle_diffs, translation_diffs = {}, {}, {}
        
        # Test model on each dataset
        for (noise, lb), (src, tgt, correct_trans) in datasets.items():
            # Perform inference and calculate loss
            pred_trans = []
            losses[(noise, lb)] = []
            for src_pcd, tgt_pcd in zip(src, tgt):
                pred = model(src_pcd)  # Adjust based on model's forward pass
                loss = mean_squared_error(tgt_pcd, pred)  # Replace with actual loss function
                losses[(noise, lb)].append(loss)
                pred_trans.append(pred)
            
            # Calculate transformation differences
            angles, translations = calculate_differences(pred_trans, correct_trans)
            angle_diffs[(noise, lb)] = angles
            translation_diffs[(noise, lb)] = translations
        
        # Plot and save results
        for metric, diffs, file_name in zip(
            ["Loss", "Angle Difference", "Translation Difference"], 
            [losses, angle_diffs, translation_diffs], 
            ["loss_plot.png", "angle_diff_plot.png", "translation_diff_plot.png"]
        ):
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.flatten()
            for i, ((noise, lb), values) in enumerate(diffs.items()):
                ax = axes[i]
                ax.hist(values, bins=20, alpha=0.7)
                ax.set_title(f"Noise: {noise}% | LB: {lb}%")
                ax.set_xlabel(metric)
                ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(model_output_dir, file_name))
            plt.close(fig)

        # Save statistics
        for metric, diffs, csv_name in zip(
            ["Angle Difference", "Translation Difference"], 
            [angle_diffs, translation_diffs], 
            ["angle_stats.csv", "translation_stats.csv"]
        ):
            stats = [
                [noise, lb, *calculate_statistics(values)]
                for (noise, lb), values in diffs.items()
            ]
            pd.DataFrame(stats, columns=["Noise", "LB", "Mean", "Stddev", "Median"]).to_csv(
                os.path.join(model_output_dir, csv_name), index=False
            )

def evaluate_models(models_dir, test_data_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.MSELoss()
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all models
    # models = {f: torch.load(os.path.join(models_dir, f)) for f in os.listdir(models_dir) if f.endswith('.pth')}
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')] 
    
    # Load test datasets
    with h5py.File(test_data_path, 'r') as test_file:
        datasets = {(noise, lb): (test_file[f"{lb}/noise_{noise}"]["src"][:], 
                                  test_file[f"{lb}/noise_{noise}"]["tgt"][:],
                                  test_file[f"{lb}/noise_{noise}"]["rotations"][:])
                    for lb in [25, 35, 45, 55]
                    for noise in [0, 1, 5, 10]}

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)

        # Initialize the model architecture
        model = FeatureMatchingModel(input_size=10683).to(device)  # Replace with your model architecture
        model.load_state_dict(torch.load(model_path))  # Load state dict
        model.eval()  # Set the model to evaluation mode
        
        # Load test datasets
        # with h5py.File(test_data_path, 'r') as test_file:
        #     datasets = {(noise, lb): (test_file[f"{lb}/noise_{noise}"]["src"][:], 
        #                               test_file[f"{lb}/noise_{noise}"]["tgt"][:],
        #                               test_file[f"{lb}/noise_{noise}"]["rotations"][:])
        #                 for lb in [25, 35, 45, 55]
        #                 for noise in [0, 1, 5, 10]}

        model_output_dir = os.path.join(output_dir, model_file.replace(".pth", ""))
        os.makedirs(model_output_dir, exist_ok=True)

        losses, angle_diffs, translation_diffs = {}, {}, {}
        
        # Test model on each dataset
        for (noise, lb), (src, tgt, correct_trans) in datasets.items():
            # Perform inference and calculate loss
            pred_trans = []
            losses[(noise, lb)] = []
            test_dataset = PointCloudDataset(src, tgt, correct_trans)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

            # for src_pcd, tgt_pcd, c_trans in zip(src, tgt, correct_trans):
                
            #     src_pcd = torch.tensor(src_pcd, dtype=torch.float32)
            #     tgt_pcd = torch.tensor(tgt_pcd, dtype=torch.float32)
            #     c_trans = torch.tensor(c_trans, dtype=torch.float32)
            #     # Replace NaN values with 0
            #     src_pcd = torch.where(torch.isnan(src_pcd), torch.tensor(0.0, dtype=torch.float32), src_pcd)
            #     tgt_pcd = torch.where(torch.isnan(tgt_pcd), torch.tensor(0.0, dtype=torch.float32), tgt_pcd)
            #     c_trans = torch.where(torch.isnan(c_trans), torch.tensor(0.0, dtype=torch.float32), c_trans)
            #     # Ensure sizes match for concatenation
            #     print(src_pcd.shape)
            #     min_size = min(src_pcd.size(0), tgt_pcd.size(0))
            #     src_pcd = src_pcd[:min_size]  # Truncate to min size
            #     tgt_pcd = tgt_pcd[:min_size]  # Truncate to min size
            c_trans_by_32 = []
            for src_pcd, tgt_pcd, c_trans in test_loader:
                # Move data to the GPU (or CPU if GPU is unavailable) and convert to Float
                src_pcd = src_pcd.to(device).float()
                tgt_pcd = tgt_pcd.to(device).float()
                c_trans = c_trans.to(device).float()

                inputs = torch.cat([src_pcd.view(src_pcd.size(0), -1), tgt_pcd.view(tgt_pcd.size(0), -1)], dim=1).to(device)
                pred = model(inputs)  # Adjust based on model's forward pass
                # print("hey", c_trans.shape)
                loss = loss_func(pred, c_trans)  # Replace with actual loss function
                losses[(noise, lb)].append(loss)
                pred_trans.append(pred)
                c_trans_by_32.append(c_trans)
            
            # Calculate transformation differences
            # angles, translations = calculate_differences(pred_trans, correct_trans)
            angles, translations = calculate_differences(pred_trans, c_trans_by_32)
            angle_diffs[(noise, lb)] = angles
            translation_diffs[(noise, lb)] = translations
        
        # Plot and save results
        for metric, diffs, file_name in zip(
            ["Loss", "Angle Difference", "Translation Difference"], 
            [losses, angle_diffs, translation_diffs], 
            ["loss_plot.png", "angle_diff_plot.png", "translation_diff_plot.png"]
        ):
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.flatten()
            for i, ((noise, lb), values) in enumerate(diffs.items()):
                ax = axes[i]
                # print(values)
                try:
                    values = [i.cpu() for i in values]
                    values = [i.detach().numpy() for i in values]
                except:
                    pass
                ax.hist(values, bins=20, alpha=0.7)
                ax.set_title(f"Noise: {noise}% | LB: {lb}%")
                ax.set_xlabel(metric)
                ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(model_output_dir, file_name))
            plt.close(fig)

        # Save statistics
        for metric, diffs, csv_name in zip(
            ["Angle Difference", "Translation Difference"], 
            [angle_diffs, translation_diffs], 
            ["angle_stats.csv", "translation_stats.csv"]
        ):
            stats = [
                [noise, lb, *calculate_statistics(values)]
                for (noise, lb), values in diffs.items()
            ]
            pd.DataFrame(stats, columns=["Noise", "LB", "Mean", "Stddev", "Median"]).to_csv(
                os.path.join(model_output_dir, csv_name), index=False
            )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Execute the evaluation
evaluate_models("feature_match_models", "test/test_data.h5", "evaluation_results")
