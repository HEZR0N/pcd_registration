import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def view_pointcloud_from_hdf5(h5_file, dataset_path, slice_index, show_both=False, src_dataset_path=None):
    """
    View a point cloud slice from an HDF5 file using Matplotlib.
    
    Parameters:
    - h5_file: str, path to the HDF5 file
    - dataset_path: str, path to the dataset within the HDF5 file
    - slice_index: int, index of the slice to visualize along the first dimension
    - show_both: bool, whether to show both source and target point clouds in the same plot
    - src_dataset_path: str, path to the source dataset, required if show_both is True
    """
    # Read the HDF5 file
    with h5py.File(h5_file, 'r') as file:
        # Access the datasets
        tgt_dataset = file[dataset_path]
        tgt_pointcloud_data = np.array(tgt_dataset)
        
        if show_both:
            # Access the source dataset if show_both is True
            if src_dataset_path is None:
                raise ValueError("Source dataset path must be provided when show_both is True.")
            src_dataset = file[src_dataset_path]
            src_pointcloud_data = np.array(src_dataset)

    # Ensure the slice_index is within bounds
    if slice_index >= tgt_pointcloud_data.shape[0]:
        raise ValueError(f"slice_index {slice_index} is out of range for dataset dimension {tgt_pointcloud_data.shape[0]}")

    # Select the slice for visualization (target point cloud)
    tgt_slice_data = tgt_pointcloud_data[slice_index, :, :]
    
    # Check the shape of the slice data
    if len(tgt_slice_data.shape) != 2 or tgt_slice_data.shape[1] != 3:
        raise ValueError("Slice data must have three columns representing x, y, z coordinates.")
    
    # Extract x, y, z coordinates for the target point cloud
    tgt_x = tgt_slice_data[:, 0]
    tgt_y = tgt_slice_data[:, 1]
    tgt_z = tgt_slice_data[:, 2]
    
    # Prepare the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot target point cloud in one color
    ax.scatter(tgt_x, tgt_y, tgt_z, s=1, c='b', label='Target')

    if show_both:
        # Ensure the slice_index is within bounds for the source dataset
        if slice_index >= src_pointcloud_data.shape[0]:
            raise ValueError(f"slice_index {slice_index} is out of range for source dataset dimension {src_pointcloud_data.shape[0]}")

        # Select the slice for visualization (source point cloud)
        src_slice_data = src_pointcloud_data[slice_index, :, :]

        # Extract x, y, z coordinates for the source point cloud
        src_x = src_slice_data[:, 0]
        src_y = src_slice_data[:, 1]
        src_z = src_slice_data[:, 2]

        # Plot source point cloud in a different color
        ax.scatter(src_x, src_y, src_z, s=1, c='r', label='Source')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Point Cloud Slice {slice_index}')
    
    # Show the legend if both point clouds are shown
    if show_both:
        ax.legend()

    # Show the plot
    plt.show()

def interactive_view(h5_file, tgt_dataset_path, show_both=False, src_dataset_path=None):
    """
    Interactive function to view slices of point cloud data from an HDF5 file.
    
    Parameters:
    - h5_file: str, path to the HDF5 file
    - tgt_dataset_path: str, path to the target dataset within the HDF5 file
    - show_both: bool, whether to show both source and target point clouds in the same plot
    - src_dataset_path: str, path to the source dataset, required if show_both is True
    """
    with h5py.File(h5_file, 'r') as file:
        dataset_shape = file[tgt_dataset_path].shape
        print(f"Dataset shape: {dataset_shape}")

    slice_index = 0
    while True:
        try:
            view_pointcloud_from_hdf5(h5_file, tgt_dataset_path, slice_index, show_both, src_dataset_path)
        except ValueError as e:
            print(e)
            break
        
        user_input = input("Enter slice index to view next (or type 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        try:
            slice_index = int(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid integer or 'q' to quit.")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="View point cloud slices from an HDF5 file.")
    
    # Adding arguments
    parser.add_argument('--train_or_test', type=str, required=True, choices=['train', 'test'], help="Specify whether to view 'train' or 'test' data.")
    parser.add_argument('--o', type=int, required=True, choices=[55, 45, 35, 25], help="Specify overlap lower bound: 55, 45, 35, 25.")
    parser.add_argument('--n', type=int, required=True, choices=[0, 1, 5, 10], help="Specify noise level: 0, 1, 5, 10.")
    parser.add_argument('--tgt_or_src', type=str, required=True, choices=['src', 'tgt'], help="Specify whether to view the 'src' or 'tgt' point cloud.")
    parser.add_argument('--show_both', action='store_true', help="Show both source and target point clouds in the same plot.")
    
    # Parse arguments
    args = parser.parse_args()

    # Construct file paths
    h5_file = f'./{args.train_or_test}/{args.train_or_test}_data.h5' 
    tgt_dataset_path = f'{args.o}/noise_{args.n}/{args.tgt_or_src}'

    # If showing both, also construct the source dataset path
    if args.show_both:
        tgt_dataset_path = f'{args.o}/noise_{args.n}/tgt'
        src_dataset_path = f'{args.o}/noise_{args.n}/src'
    else:
        src_dataset_path = None

    # Start the interactive viewer
    interactive_view(h5_file, tgt_dataset_path, args.show_both, src_dataset_path)

if __name__ == "__main__":
    main()
