import os
import shutil
import pandas as pd
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

# Create folders
def create_folders(dataset, folder_name):
    path = os.path.join(TARGET_PATH, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    for category in dataset['Type'].unique():
        category_path = os.path.join(path, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

# Copy image data from the original path to the target path
def copy_files(dataset, folder_name):
    for _, row in tqdm(dataset.iterrows()):
        source_path = os.path.join(IMAGE_PATH, row['ImageName'])
        target_path = os.path.join(TARGET_PATH, folder_name, row['Type'], row['ImageName'])
        shutil.copy(source_path, target_path)

if __name__ == '__main__':
    # Read YAML file to get the config
    with open('../00_config.yml', 'r') as file:
        config = yaml.safe_load(file)
    config = edict(config)

    # Read original image data names and labels
    data = pd.read_csv('data.csv', dtype=str)

    # Build two dataset for training and testing
    for proximal in ['first', 'middle']:
        IMAGE_PATH = config.PATH.IMAGE_PATH
        IMAGE_PATH = os.path.join(IMAGE_PATH, f'2D_{proximal}', 'data')
        print('IMAGE_PATH: ', IMAGE_PATH)

        TARGET_PATH = config.PATH.DATASET_TARGET_PATH
        TARGET_PATH = os.path.join(TARGET_PATH, 'Cross_Validation', proximal)
        print('TARGET_PATH: ', TARGET_PATH)

        # Split data into train and test (because we use cross-validation, here do not have validation dataset)
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        grouped = data.groupby('Type')
        for _, group in grouped:
            train_group, test_group = train_test_split(group, test_size=0.2, random_state=2024)
            train_data = pd.concat([train_data, train_group])
            test_data = pd.concat([test_data, test_group])

        create_folders(train_data, 'train')
        create_folders(test_data, 'test')

        copy_files(train_data, 'train')
        copy_files(test_data, 'test')