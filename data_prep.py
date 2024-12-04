import os
import gdown
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


original_train_data_path = os.path.join('original_train_data', 'MVP_Train_RG.h5')
original_test_data_path = os.path.join('original_test_data', 'MVP_Test_RG.h5')
train_data_path = os.path.join('train', 'train_data.h5')
test_data_path = os.path.join('test', 'test_data.h5')

# Download the original data
def download_original_data(is_train='train'):
    print('Downloading data...')
    # Create the directory if it doesn't exist
    os.makedirs('original_train_data', exist_ok=True)
    os.makedirs('original_test_data', exist_ok=True)

    # define data url - it's in google drive
    original_train_data_file_id = '1k6JRRKY-_a9oTBS4M-DUMxG76SyJOjbs'
    train_url = f'https://drive.google.com/uc?id={original_train_data_file_id}'
    original_test_data_file_id = '16jSMsYZOI8QU-g0_TgqWbxzuzauSKCBQ'
    test_url = f'https://drive.google.com/uc?id={original_test_data_file_id}'

    # Check if files already exist
    if not os.path.exists(original_train_data_path):
        gdown.download(train_url, original_train_data_path, quiet=False)
    else:
        print('Train data already exists, skipping download.')
    if not os.path.exists(original_test_data_path):
        gdown.download(test_url, original_test_data_path, quiet=False)
    else:
        print('Test data already exists, skipping download.')

def random_rotation(point, angle_degrees, rand_axis):
    # Generate a random rotation vector
    random_axis = rand_axis
    random_axis /= np.linalg.norm(random_axis)  # Normalize

    # Create a rotation object for a rotation around the random axis
    rotation = R.from_rotvec(random_axis * np.radians(angle_degrees))

    # Rotate the point
    rotated_point = rotation.apply(point)
    return rotated_point

# Function to apply random rotation and translation to a point
def random_rotation_and_translation(point, angle_degrees, rand_axis, translation_vector):
    # Apply rotation using the existing random_rotation function
    rotated_point = random_rotation(point, angle_degrees, rand_axis)
    
    # Apply translation by adding the translation vector to the rotated point
    translated_point = rotated_point + translation_vector
    
    return translated_point

# Curate datasets from the original data
def curate_datasets():
    print('Curating datasets...')
    # Create the directory if it doesn't exist
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    default_upper_bound = 0.85

    for original_data_path, data_path in [(original_train_data_path, train_data_path),(original_test_data_path, test_data_path)]:
        # Read the HDF5 file
        with h5py.File(original_data_path, 'r') as file:
            # Access the dataset
            dataset = file['complete']
            # Read data into a NumPy array
            orginal_pointcloud_data = np.array(dataset)
            # print(orginal_pointcloud_data.shape)
            # orginal_pointcloud_data = orginal_pointcloud_data[:4,:,:]
            # print(orginal_pointcloud_data.shape)

        # Generate random noise between -X% and +X% for each XYZ value
        get_noise_matrix = lambda x: np.random.uniform(-x / 100, x / 100, orginal_pointcloud_data.shape)
        add_noise = lambda x: orginal_pointcloud_data * (1 + get_noise_matrix(x))

        # Save the point cloud data to HDF5 file
        with h5py.File(data_path, 'w') as new_file:
            # curate datasets with different percents of overalapping points
            for lower_bound in [55, 45, 35, 25]:
            # for lower_bound in [55]:
                # no_noise = np.copy(orginal_pointcloud_data)
                # Add noise to the original point cloud data
                # noise_level_1 = add_noise(0.5)
                # noise_level_5 = add_noise(2.5)
                # noise_level_10 = add_noise(5)
                
                # Determine random number of points that will overlap
                # Get a percent between the upper and lower bound
                overlap_percent = np.round(np.random.uniform(lower_bound/100, default_upper_bound, orginal_pointcloud_data.shape[0]), 2)
                # Calculate the number of points that will be the same in both taget and source pcds    
                total_points = orginal_pointcloud_data.shape[1]
                # num_overlap_points = int(total_points * (1 / ((200 / (overlap_percent * 100)) - 1)))
                # num_overlap_points = int(total_points / (2 / overlap_percent - 1))
                num_overlap_points = np.int64(total_points * overlap_percent / (2 - overlap_percent))

                # Randomly select indices
                # create orginal_pointcloud_data.shape[0] (num of pcds) list of randomly ordered indices
                random_indices = np.array([np.random.choice(total_points, total_points, replace=False) for i in range(orginal_pointcloud_data.shape[0])])

                # Create src and tgt with random points
                for noise_level in [0, 1, 5, 10]:
                # for noise_level in [0]:
                    # Add noise to the original point cloud data
                    cur_dataset = add_noise(noise_level / 2)
                    # Copy dataset[random_indices[:num_overlap_points]] to target and source
                    # Copy dataset[random_indices[num_overlap_points:... first half]] to source 
                    # Copy dataset[random_indices[num_overlap_points:... second half]] to target
                    # print(orginal_pointcloud_data)
                    # print(random_indices[:num_overlap_points + (total_points-num_overlap_points)//2])
                    # print(random_indices)
                    # print(num_overlap_points + (total_points - num_overlap_points) // 2)
                    # list of orginal_pointcloud_data.shape[0] indexes to split source and target data
                    split_list = num_overlap_points + (total_points - num_overlap_points) // 2
                    # print([row[:index] for row, index in zip(random_indices, split_list)])
                    source_rand_indices = [row[:src_tgt_index] for row, src_tgt_index in zip(random_indices, split_list)]
                    target_rand_indices = [np.concatenate((row[:overlap], row[src_tgt_index:])) for row, overlap, src_tgt_index in zip(random_indices, num_overlap_points, split_list)]

                    # source_pcds = cur_dataset[random_indices[:split_list]]
                    # source_pcds = cur_dataset[:, source_rand_indices[0]]
                    source_pcds = [row[rand_indices] for row, rand_indices in zip(cur_dataset, source_rand_indices)]
                    # target_pcds = cur_dataset[np.concatenate((random_indices[:num_overlap_points], random_indices[split_list:]))]
                    target_pcds = [row[rand_indices] for row, rand_indices in zip(cur_dataset, target_rand_indices)]
                    # print(source_pcds)
                    max_length = max(len(row) for row in source_pcds)
                    # print(max_length)
                    # print(source_pcds[0])
                    # print(np.pad(source_pcds[2], (0, max_length - len(source_pcds[2])), mode='constant', constant_values=np.nan))
                    # print([np.pad(row, (0, max_length - len(row)), mode='constant', constant_values=np.nan) for row in source_pcds])
                    # source_pcds = np.array([np.pad(row, (0, max_length - len(row)), mode='constant', constant_values=np.nan) for row in source_pcds])
                    # print(len(source_pcds))
                    # for row in source_pcds:
                    #     print(len(np.pad(row, (0, max_length - len(row)), mode='constant', constant_values=np.nan)))
                    #     for i in row:
                    #         if len(i) < 3:
                    #             print(len(i))
                    # Determine the maximum length for each "slice"
                    max_source_rows = max(len(layer) for layer in source_pcds)
                    max_source_cols = max(len(array) for layer in source_pcds for array in layer)
                    max_target_rows = max(len(layer) for layer in target_pcds)
                    max_target_cols = max(len(array) for layer in target_pcds for array in layer)
                    # padded_source_pcds = np.full((len(source_pcds), max_length), np.nan)
                    padded_source_pcds = np.full((len(source_pcds), max_source_rows, max_source_cols), np.nan)
                    padded_target_pcds = np.full((len(target_pcds), max_target_rows, max_target_cols), np.nan)
                    # Fill the padded array
                    # for i, row in enumerate(source_pcds):
                        # padded_source_pcds[i, :len(row)] = row  # Only fill the non-padded part
                    transformations = []
                    # each layer is a whole pointcloud
                    # each row is a single point
                    for i, layer in enumerate(source_pcds):
                        for j, row in enumerate(layer):
                            padded_source_pcds[i, j, :len(row)] = row  # Fill with the row values
                    for i, layer in enumerate(target_pcds):
                        # define rotation/translation
                        rand_axis = np.random.rand(3)
                        angle = np.random.uniform(0, 360)  # Random angle between 0 and 360 degrees
                        rand_axis /= np.linalg.norm(rand_axis)  # Normalize axis
                        # print(rand_axis.append(angle))
                        # Generate a random translation vector (can be within a certain range)
                        translation_vector = np.random.uniform(-1, 1, 3)  # Random translation in each direction (-1 to 1)
                        rot_and_angle = rand_axis[:]
                        rot_and_angle = np.append(rot_and_angle, angle)
                        rot_and_trans = np.append(rot_and_angle, translation_vector)  # Append translation vector
                        # print(rot_and_angle)
                        transformations.append(rot_and_trans)
                        # transformations.append([rand_axis, angle])
                        # transformations.append(rand_axis)
                        for j, row in enumerate(layer):
                            # ... = row * rotation/translation
                            # padded_target_pcds[i, j, :len(row)] = row  # Fill with the row values
                            # padded_target_pcds[i, j, :len(row)] = row
                            # continue
                            padded_target_pcds[i, j, :len(row)] = random_rotation_and_translation(row, angle, rand_axis, translation_vector)  # Fill with the row values

                    # print(padded_source_pcds)
                    # source_pcds = np.array([np.pad(row, (0, max_length - len(row)), mode='constant', constant_values=np.nan) for row in source_pcds])
                    new_file.create_dataset(f'{lower_bound}/noise_{noise_level}/src', data=padded_source_pcds)
                    # new_file.create_dataset(f'{lower_bound}/noise_{noise_level}/src', data=source_pcds)
                    # print(f'Saved Training data for +/-{noise_level/2}% noise and lower overlap bound {lower_bound} to {data_path}')
                    # new_file.create_dataset(f'{lower_bound}/noise_{noise_level}/tgt', data=target_pcds)
                    new_file.create_dataset(f'{lower_bound}/noise_{noise_level}/tgt', data=padded_target_pcds)
                    # Save transformations
                    new_file.create_dataset(f'{lower_bound}/noise_{noise_level}/transformations', data=transformations)
                    print(f'Saved data for +/-{noise_level/2}% noise and lower overlap bound {lower_bound}% to {data_path}')

    

    # print(f"Point cloud data with +/-{0.5}% noise saved to {train_data_path}")

    # os.rmdir('original_train_data')
    # os.rmdir('original_test_data')


# Download the original data
download_original_data()
# Curate datasets from the original data
curate_datasets()

"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def add_noise_and_save(h5_file, dataset_path, output_file):
    '''
    Add noise to the XYZ values of the point cloud data and save to a new HDF5 file.
    
    Parameters:
    - h5_file: str, path to the original HDF5 file
    - dataset_path: str, path to the dataset within the HDF5 file
    - output_file: str, path for the new HDF5 file to save the noisy data
    '''
    # Read the original HDF5 file
    with h5py.File(h5_file, 'r') as file:
        dataset = file[dataset_path]
        pointcloud_data = np.array(dataset)
    
    # Generate random noise between -0.5% and +0.5% for each XYZ value
    noise_factor = 0.005  # 0.5% as a decimal
    noise = np.random.uniform(-noise_factor, noise_factor, pointcloud_data.shape)
    
    # Add noise to the original point cloud data
    noisy_pointcloud_data = pointcloud_data * (1 + noise)
    
    # Save the noisy point cloud data to a new HDF5 file
    with h5py.File(output_file, 'w') as new_file:
        new_file.create_dataset('noisy_pcds', data=noisy_pointcloud_data)
    
    print(f"Noisy point cloud data saved to {output_file}")

def view_pointcloud_from_hdf5(h5_file, dataset_path, slice_index):
    # ... (existing function remains unchanged)

def interactive_view(h5_file, dataset_path):
    # ... (existing function remains unchanged)

# Example usage
h5_file = 'MVP_Test_CP.h5'
dataset_path = 'complete_pcds'
output_file = 'noise_1.h5'

# Add noise and save to new file
add_noise_and_save(h5_file, dataset_path, output_file)

# Start interactive viewing from the original file
interactive_view(h5_file, dataset_path)

"""