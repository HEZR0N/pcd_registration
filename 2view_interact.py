import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def view_pointcloud_from_hdf5(h5_file, dataset_path, slice_index):
    """
    View a point cloud slice from an HDF5 file using Matplotlib.
    
    Parameters:
    - h5_file: str, path to the HDF5 file
    - dataset_path: str, path to the dataset within the HDF5 file
    - slice_index: int, index of the slice to visualize along the first dimension
    """
    # Read the HDF5 file
    with h5py.File(h5_file, 'r') as file:
        # Access the dataset
        dataset = file[dataset_path]
        # Read data into a NumPy array
        pointcloud_data = np.array(dataset)
    
    # Ensure the slice_index is within bounds
    if slice_index >= pointcloud_data.shape[0]:
        raise ValueError(f"slice_index {slice_index} is out of range for dataset dimension {pointcloud_data.shape[0]}")
    
    # Select the slice for visualization
    slice_data = pointcloud_data[slice_index, :, :]
    
    # Check the shape of the slice data
    if len(slice_data.shape) != 2 or slice_data.shape[1] != 3:
        raise ValueError("Slice data must have three columns representing x, y, z coordinates.")
    
    # Extract x, y, z coordinates
    x = slice_data[:, 0]
    y = slice_data[:, 1]
    z = slice_data[:, 2]
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1)  # s is the marker size

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Point Cloud Slice {slice_index}')

    plt.show()

def interactive_view(h5_file, dataset_path):
    """
    Interactive function to view slices of point cloud data from an HDF5 file.
    
    Parameters:
    - h5_file: str, path to the HDF5 file
    - dataset_path: str, path to the dataset within the HDF5 file
    """
    with h5py.File(h5_file, 'r') as file:
        dataset_shape = file[dataset_path].shape
        print(f"Dataset shape: {dataset_shape}")

    slice_index = 0
    while True:
        try:
            view_pointcloud_from_hdf5(h5_file, dataset_path, slice_index)
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

# Example usage
# h5_file = 'MVP_Test_CP.h5'
# dataset_path = 'complete_pcds'
h5_file = 'Point_Cloud_Registration_Hezron_Perez/BAK/train/train_data.h5'
dataset_path = '55/noise_0/tgt'

interactive_view(h5_file, dataset_path)
