import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from pyntcloud import PyntCloud
from sklearn.preprocessing import StandardScaler

def plot_accuracy_for_all_combinations(angle_diffs, translation_diffs, dataset_name, noise_levels, lower_bounds):
    # Create a new figure for angle differences
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))  # 4x4 grid of subplots
    axes = axes.flatten()
    
    plot_idx = 0
    for noise_level in noise_levels:
        for lower_bound in lower_bounds:
            ax = axes[plot_idx]
            angle_diff = angle_diffs[(noise_level, lower_bound)]
            translation_diff = translation_diffs[(noise_level, lower_bound)]
            
            # Plot angle differences
            ax.hist(angle_diff, bins=20, color='blue', alpha=0.7, label="Angle Diff")
            ax.set_title(f"Noise: {noise_level}% | LB: {lower_bound}%")
            ax.set_xlabel("Angle Difference (degrees)")
            ax.set_ylabel("Frequency")
            ax.legend()

            plot_idx += 1

    plt.tight_layout()
    # Save the angle difference plot as PNG
    plt.savefig(f"{dataset_name}_angle_accuracy_plots.png", dpi=300)
    # plt.show()

    # Create a new figure for translation differences
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))  # 4x4 grid of subplots
    axes = axes.flatten()
    
    plot_idx = 0
    for noise_level in noise_levels:
        for lower_bound in lower_bounds:
            ax = axes[plot_idx]
            translation_diff = translation_diffs[(noise_level, lower_bound)]
            
            # Plot translation differences
            ax.hist(translation_diff, bins=20, color='green', alpha=0.7, label="Translation Diff")
            ax.set_title(f"Noise: {noise_level}% | LB: {lower_bound}%")
            ax.set_xlabel("Translation Difference (units)")
            ax.set_ylabel("Frequency")
            ax.legend()

            plot_idx += 1

    plt.tight_layout()
    # Save the translation difference plot as PNG
    plt.savefig(f"{dataset_name}_translation_accuracy_plots.png", dpi=300)
    # plt.show()

def calculate_statistics(diffs):
    """Calculate mean, stddev, and median for a list of differences"""
    mean = np.mean(diffs)
    stddev = np.std(diffs)
    median = np.median(diffs)
    return mean, stddev, median

# Function to compute ICP (Iterative Closest Point) between two point clouds
def icp(source_pcd, target_pcd, max_iterations=100, tolerance=1e-5):
    target_pcd = target_pcd[~np.isnan(target_pcd).all(axis=1)]
    source_pcd = source_pcd[~np.isnan(source_pcd).all(axis=1)]
    
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_pcd)
    
    prev_error = float('inf')
    transformation = np.eye(4)

    for _ in range(max_iterations):
        # Find closest points from the target to the source
        distances, indices = nn.kneighbors(source_pcd)

        # Compute the centroid
        source_centroid = np.mean(source_pcd, axis=0)
        target_centroid = np.mean(target_pcd[indices.flatten()], axis=0)

        # Compute covariance matrix
        H = np.dot((source_pcd - source_centroid).T, target_pcd[indices.flatten()] - target_centroid)

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(H)
        R_matrix = np.dot(Vt.T, U.T)

        # Compute translation
        t_vector = target_centroid - np.dot(source_centroid, R_matrix)

        # Update transformation matrix
        transformation[:3, :3] = R_matrix
        transformation[:3, 3] = t_vector

        # Apply transformation to the source point cloud
        transformed_source_pcd = np.dot(source_pcd, R_matrix.T) + t_vector

        # Compute the error as the sum of squared distances
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    return transformation, transformed_source_pcd

# Function to calculate the transformation differences
def calculate_differences(predicted_transformations, correct_transformations):
    angle_diffs = []
    translation_diffs = []

    for pred_trans, correct_trans in zip(predicted_transformations, correct_transformations):
        pred_rotation_matrix = pred_trans[:3, :3]
        pred_translation = pred_trans[:3, 3]

        axis_angle = correct_trans[:4]
        correct_translation = correct_trans[4:]
        
        correct_rotation = R.from_rotvec(axis_angle[:3] * np.radians(axis_angle[3]))  # Axis-angle to rotation matrix
        correct_rotation_matrix = correct_rotation.as_matrix()

        # Compute angle difference
        angle_diff = np.degrees(np.arccos((np.trace(np.dot(correct_rotation_matrix.T, pred_rotation_matrix)) - 1) / 2))
        translation_diff = np.linalg.norm(pred_translation - correct_translation)

        angle_diffs.append(angle_diff)
        translation_diffs.append(translation_diff)

    return angle_diffs, translation_diffs

# Function to process the data and compute ICP
def process_data(data_path, dataset_name):
    output_dir = "ICP"
    os.makedirs(output_dir, exist_ok=True)  # Create folder if not exists

    with h5py.File(data_path, 'r') as file:
        # Iterate through different noise levels and lower bounds
        noise_levels = [0, 1, 5, 10]
        lower_bounds = [55, 45, 35, 25]

        angle_diffs = {}
        translation_diffs = {}

        transformed_source_pcds = []
        target_pcds = []

        for lower_bound in lower_bounds:
            for noise_level in noise_levels:
                # Get source and target point clouds
                src = file[f'{lower_bound}/noise_{noise_level}/src'][:]
                tgt = file[f'{lower_bound}/noise_{noise_level}/tgt'][:]
                correct_transformations = file[f'{lower_bound}/noise_{noise_level}/rotations'][:]

                predicted_transformations = []
                for source, target in zip(src, tgt):
                    # Run ICP to get the predicted transformation
                    transformation, transformed_source_pcd = icp(source, target)
                    predicted_transformations.append(transformation)
                    
                    # Save the transformed source and target point clouds
                    transformed_source_pcds.append(transformed_source_pcd)
                    target_pcds.append(target)

                # Calculate angle and translation differences
                angle_diff, translation_diff = calculate_differences(predicted_transformations, correct_transformations)
                angle_diffs[(noise_level, lower_bound)] = angle_diff
                translation_diffs[(noise_level, lower_bound)] = translation_diff

        # Plot and save results
        plot_accuracy_for_all_combinations(angle_diffs, translation_diffs, dataset_name, noise_levels, lower_bounds)

        # Calculate and save statistics
        angle_stats = []
        translation_stats = []
        
        for noise_level in noise_levels:
            for lower_bound in lower_bounds:
                angle_diff = angle_diffs[(noise_level, lower_bound)]
                translation_diff = translation_diffs[(noise_level, lower_bound)]

                # Calculate statistics
                angle_mean, angle_stddev, angle_median = calculate_statistics(angle_diff)
                translation_mean, translation_stddev, translation_median = calculate_statistics(translation_diff)

                # Save to lists
                angle_stats.append([noise_level, lower_bound, angle_mean, angle_stddev, angle_median])
                translation_stats.append([noise_level, lower_bound, translation_mean, translation_stddev, translation_median])

                # Print statistics
                print(f"Noise: {noise_level}% | LB: {lower_bound}%")
                print(f"Angle - Mean: {angle_mean:.2f}, Stddev: {angle_stddev:.2f}, Median: {angle_median:.2f}")
                print(f"Translation - Mean: {translation_mean:.2f}, Stddev: {translation_stddev:.2f}, Median: {translation_median:.2f}")
                print("-" * 50)

        # Save statistics to CSV files
        angle_df = pd.DataFrame(angle_stats, columns=["Noise Level", "Lower Bound", "Mean", "Stddev", "Median"])
        translation_df = pd.DataFrame(translation_stats, columns=["Noise Level", "Lower Bound", "Mean", "Stddev", "Median"])
        
        angle_df.to_csv(os.path.join(output_dir, f"{dataset_name}_angle_stats.csv"), index=False)
        translation_df.to_csv(os.path.join(output_dir, f"{dataset_name}_translation_stats.csv"), index=False)

        # Pad Tranformed Source Pointclouds
        max_transformed_source_rows = max(len(layer) for layer in transformed_source_pcds)
        max_transformed_source_cols = max(len(array) for layer in transformed_source_pcds for array in layer)
        padded_transformed_source_pcds = np.full((len(transformed_source_pcds), max_transformed_source_rows, max_transformed_source_cols), np.nan)
        # each layer is a whole pointcloud
        # each row is a single point
        for i, layer in enumerate(transformed_source_pcds):
            print(i)
            if i > 50:
                break
            for j, row in enumerate(layer):
                padded_transformed_source_pcds[i, j, :len(row)] = row  # Fill with the row values
        # Save the transformed source and target point clouds to a new HDF5 file
        with h5py.File(os.path.join(output_dir, f"icp_{dataset_name.lower()}.h5"), 'w') as new_file:
            new_file.create_dataset('transformed_source_pcd', data=np.array(padded_transformed_source_pcds))
            # new_file.create_dataset('transformed_source_pcd', data=np.array(transformed_source_pcds))
            new_file.create_dataset('target_pcd', data=np.array(target_pcds))


def process_data2(data_path, dataset_name):
    output_dir = "ICP"
    os.makedirs(output_dir, exist_ok=True)  # Create folder if not exists

    with h5py.File(data_path, 'r') as file:
        # Iterate through different noise levels and lower bounds
        noise_levels = [0, 1, 5, 10]
        lower_bounds = [55, 45, 35, 25]

        angle_diffs = {}
        translation_diffs = {}

        for lower_bound in lower_bounds:
            for noise_level in noise_levels:
                # Get source and target point clouds
                src = file[f'{lower_bound}/noise_{noise_level}/src'][:]
                tgt = file[f'{lower_bound}/noise_{noise_level}/tgt'][:]
                correct_transformations = file[f'{lower_bound}/noise_{noise_level}/rotations'][:]

                predicted_transformations = []
                transformed_source_pcds = []
                target_pcds = []

                for source, target in zip(src, tgt):
                    # Run ICP to get the predicted transformation
                    transformation, transformed_source_pcd = icp(source, target)
                    predicted_transformations.append(transformation)

                    # Save the transformed source and target point clouds
                    transformed_source_pcds.append(transformed_source_pcd)
                    target_pcds.append(target)

                # Calculate angle and translation differences
                angle_diff, translation_diff = calculate_differences(predicted_transformations, correct_transformations)
                angle_diffs[(noise_level, lower_bound)] = angle_diff
                translation_diffs[(noise_level, lower_bound)] = translation_diff

        # Plot and save results
        plot_accuracy_for_all_combinations(angle_diffs, translation_diffs, dataset_name, noise_levels, lower_bounds)

        # Calculate and save statistics
        angle_stats = []
        translation_stats = []

        for noise_level in noise_levels:
            for lower_bound in lower_bounds:
                angle_diff = angle_diffs[(noise_level, lower_bound)]
                translation_diff = translation_diffs[(noise_level, lower_bound)]

                # Calculate statistics
                angle_mean, angle_stddev, angle_median = calculate_statistics(angle_diff)
                translation_mean, translation_stddev, translation_median = calculate_statistics(translation_diff)

                # Save to lists
                angle_stats.append([noise_level, lower_bound, angle_mean, angle_stddev, angle_median])
                translation_stats.append([noise_level, lower_bound, translation_mean, translation_stddev, translation_median])

                # Print statistics
                print(f"Noise: {noise_level}% | LB: {lower_bound}%")
                print(f"Angle - Mean: {angle_mean:.2f}, Stddev: {angle_stddev:.2f}, Median: {angle_median:.2f}")
                print(f"Translation - Mean: {translation_mean:.2f}, Stddev: {translation_stddev:.2f}, Median: {translation_median:.2f}")
                print("-" * 50)

        # Save statistics to CSV files
        angle_df = pd.DataFrame(angle_stats, columns=["Noise Level", "Lower Bound", "Mean", "Stddev", "Median"])
        translation_df = pd.DataFrame(translation_stats, columns=["Noise Level", "Lower Bound", "Mean", "Stddev", "Median"])

        angle_df.to_csv(os.path.join(output_dir, f"{dataset_name}_angle_stats.csv"), index=False)
        translation_df.to_csv(os.path.join(output_dir, f"{dataset_name}_translation_stats.csv"), index=False)

        # Save the transformed source and target point clouds to a new HDF5 file
        with h5py.File(os.path.join(output_dir, f"icp_{dataset_name.lower()}.h5"), 'w') as new_file:
            for lower_bound in lower_bounds:
                for noise_level in noise_levels:
                    # Generate dataset paths
                    transformed_src_path = f"{lower_bound}/noise_{noise_level}/transformed_src"
                    target_path = f"{lower_bound}/noise_{noise_level}/target"

                    # Filter the corresponding transformed source and target point clouds
                    group_transformed_src = []
                    group_targets = []

                    src = file[f'{lower_bound}/noise_{noise_level}/src'][:]
                    tgt = file[f'{lower_bound}/noise_{noise_level}/tgt'][:]
                    
                    for i, source in enumerate(src):
                        if i > 50:
                            break
                        _, transformed_source_pcd = icp(source, tgt[i])
                        group_transformed_src.append(transformed_source_pcd)
                        group_targets.append(tgt[i])

                    # Convert lists to padded arrays (handle point cloud sizes)
                    max_src_rows = max(len(layer) for layer in group_transformed_src)
                    max_src_cols = max(len(array) for layer in group_transformed_src for array in layer)
                    padded_transformed_src = np.full((len(group_transformed_src), max_src_rows, max_src_cols), np.nan)

                    for i, layer in enumerate(group_transformed_src):
                        if i > 50:
                            break
                        for j, row in enumerate(layer):
                            padded_transformed_src[i, j, :len(row)] = row

                    # Save datasets in the HDF5 file
                    new_file.create_dataset(transformed_src_path, data=padded_transformed_src)
                    new_file.create_dataset(target_path, data=np.array(group_targets))




# Process the training and testing data
# process_data('train/train_data.h5', 'Training Data')
process_data2('test/test_data.h5', 'Testing Data')
