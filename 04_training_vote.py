import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import statistics
import pickle
import warnings
warnings.filterwarnings("ignore")
import yaml
from easydict import EasyDict as edict
import argparse

###################################################
parser = argparse.ArgumentParser()
parser.add_argument('--proximal', type=str, default='middle')
parser.add_argument('--fold', type=int, default=5)
args = parser.parse_args()
print('\n\n', args, '\n')

# Read config to get the path
with open('./00_config.yml', 'r') as file:
    config = yaml.safe_load(file)
config = edict(config)

# Get Data path
proximal = args.proximal
data_dir = os.path.join(config.PATH.DATASET_TARGET_PATH[1:], 'Cross_Validation', proximal)

save_dir = os.path.join('../output/vote', proximal)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Build Data Loader
base_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),] + base_transform.transforms)
val_transform = base_transform

test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=base_transform)
test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)

custom_order = ['T1', 'T2', 'FLAIR', 'DWI', 'DTI', 'OTHER']
class_names = test_dataset.classes
num_classes =  len(class_names)


def find_files(root_folder, filename):
    found_files = [] 
    for root, dirs, files in os.walk(root_folder):
        if filename in files:
            found_files.append(os.path.join(root, filename))
    return found_files if found_files else "No files found."

def calculate_vote(row):
    most_common = row.iloc[1:].mode()
    if len(most_common) > 1:
        return row['Ground Truth']
    return most_common.iloc[0]

if __name__ == '__main__':
    res = {}

    # Get ground truth
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            labels = labels.to(device)
            labels_mapped = [custom_order.index(class_names[label]) for label in labels.cpu().numpy()]
            all_labels.extend(labels_mapped)
    res['Ground Truth'] = all_labels

    storage = []
    # for n_splits in [10]:
    n_splits = args.fold

    root_folder = f'./output/{n_splits}-Fold'
    results = []

    for n in range(n_splits):
        # Read best model
        if proximal == 'middle':
            filename = f'Folder{n}_middle_best_model.pth'
        else:
            filename = f'Folder{n}_first_best_model.pth'
        file_paths = find_files(root_folder, filename)

        for file_path in file_paths:
            model_name = file_path.split('/')[-2]
            print(model_name)

            if model_name == 'alexnet':
                model = models.alexnet()
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'googlenet':
                # model = models.googlenet()
                # num_ftrs = model.fc.in_features
                # model.fc = torch.nn.Linear(num_ftrs, num_classes)
                continue
            elif model_name == 'resnet18':
                model = models.resnet18()
                num_ftrs = model.fc.in_features
                model.fc = torch.nn.Linear(num_ftrs, num_classes)
            elif model_name == 'densenet121':
                model = models.densenet121()
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'efficientnet_b0':
                model = models.efficientnet_b0()
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'convnext_tiny':
                model = models.convnext_tiny()
                num_ftrs = model.classifier[2].in_features
                model.classifier[2] = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'efficientnet_v2_s':
                model = models.efficientnet_v2_s()
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'mobilenet_v3_small':
                model = models.mobilenet_v3_small()
                num_ftrs = model.classifier[3].in_features
                model.classifier[3] = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'vgg11':
                model = models.vgg11()
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs, num_classes)

            model = model.to(device)
            model.load_state_dict(torch.load(file_path, map_location=device), strict=True)

            model.eval()
            all_preds = []

            with torch.no_grad():
                for images, labels in tqdm(test_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    
                    preds_mapped = [custom_order.index(class_names[pred]) for pred in preds.cpu().numpy()]
                    all_preds.extend(preds_mapped)

            res[model_name] = all_preds
            accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
            print(f'Accuracy on test set: {100 * accuracy:.2f}%')

        df = pd.DataFrame(res)
        # df.drop('googlenet', axis=1, inplace=True)

        # if model_name == 'googlenet':
        #     file_dir = os.path.join('../output/', f'{n_splits}-Fold', 'googlenet', f'Folder{n}_first_test_labels_and_preds.pkl')
        #     with open(file_dir, 'rb') as file:
        #         googlenet_result = pickle.load(file)
        #     df['googlenet'] = pd.DataFrame(googlenet_result)['all_preds']

        df['vote'] = df.apply(calculate_vote, axis=1)

        accuracy = (df['Ground Truth'] == df['vote']).mean()
        print('\n##################################')
        print(f"{n_splits}-Fold, {n}, Accuracy: {accuracy:.2%}")
        print('##################################/n')
        save_path = os.path.join(save_dir, f'{n_splits}-Fold-{n}.csv')
        df.to_csv(save_path, index=False)
        results.append(float(accuracy))

    mean_value = statistics.mean(results)
    std_dev = statistics.stdev(results)

    print('\n##################################')
    print(results)
    print("Mean:", mean_value)
    print("Standard Deviation:", std_dev)
    print('##################################/n')
    storage.append(results)
    storage.append(mean_value)
    storage.append(std_dev)
    
    print(storage)
