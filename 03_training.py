# Usage:
# nohup python -u training.py --proximal middle --model alexnet --epoch 100 --fold 5 > ./AlexNet/output_mid.log 2>&1 &
# nohup python -u training.py --proximal middle --model googlenet --epoch 100 --fold 5 > ./GoogLeNet/output_mid.log 2>&1 &
# nohup python -u training.py --proximal middle --model resnet18 --epoch 100 --fold 5 > ./ResNet18/output_mid.log 2>&1 &
# nohup python -u training.py --proximal middle --model densenet121 --epoch 100 --fold 5 > ./DenseNet121/output_mid.log 2>&1 &

import torch
import torch.nn as nn
import torch.optim as optim

import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models

import numpy as np
from sklearn.model_selection import KFold
import statistics
import argparse
import pickle
import yaml
from easydict import EasyDict as edict
# import functools
# print = functools.partial(print, flush=True)

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)

###################################################
parser = argparse.ArgumentParser()
parser.add_argument('--proximal', type=str, default='middle')
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--fold', type=int, default=5)
args = parser.parse_args()
print('\n\n', args, '\n')

# Read config to get the path
with open('../00_config.yml', 'r') as file:
    config = yaml.safe_load(file)
config = edict(config)

num_epochs = args.epoch
FILE_NAME = args.proximal

# Get Data path
DATA_PATH = os.path.join(config.PATH.IMAGE_PATH, 'Cross_Validation', args.proximal)
print('DATA_PATH:', DATA_PATH)
# Saving model path
BEST_MODLE_PATH = os.path.join('./output', f'{args.fold}-Fold', args.model)
print('BEST_MODLE_PATH:', BEST_MODLE_PATH)

if not os.path.exists(BEST_MODLE_PATH):
    os.makedirs(BEST_MODLE_PATH)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, '\n')

# K Fold Cross Validation
k_folds = args.fold
kfold = KFold(n_splits=k_folds, shuffle=True)

# Build Data Loader
base_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),] + base_transform.transforms)
## Training data
dataset = datasets.ImageFolder(root=f'{DATA_PATH}/train', transform=base_transform)
## Testing data
test_dataset = datasets.ImageFolder(root=f'{DATA_PATH}/test', transform=base_transform)
test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)

class_names = test_dataset.classes
num_classes =  len(class_names)

if __name__ == '__main__':
    results = []
    criterion = nn.CrossEntropyLoss()

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Build dataloader
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(test_ids)
        
        train_dataset = datasets.ImageFolder(root=f'{DATA_PATH}/train', transform=train_transform)
        val_dataset = datasets.ImageFolder(root=f'{DATA_PATH}/train', transform=base_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_subsampler)
        val_loader = DataLoader(val_dataset, batch_size=4, sampler=val_subsampler)

        samples_per_phase = {'train': len(train_subsampler), 'val': len(val_subsampler)}
        dataloaders = {'train': train_loader, 'val': val_loader}

        # Build model
        if args.model == 'alexnet':
            model = models.alexnet(pretrained=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        elif args.model == 'googlenet':
            model = models.googlenet(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif args.model == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif args.model == 'densenet121':
            model = models.densenet121(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif args.model == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif args.model == 'convnext_tiny':
            model = models.convnext_tiny(pretrained=True)
            num_ftrs = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        elif args.model == 'efficientnet_v2_s':
            model = models.efficientnet_v2_s(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif args.model == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=True)
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        elif args.model == 'vgg11':
            model = models.vgg11(pretrained=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Start Training
        for epoch in range(num_epochs): 
            best_acc = 0.0
            best_epoch_num = 0
            print(f'\nFold {fold}, Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()

                epoch_loss = running_loss / samples_per_phase[phase]
                epoch_acc = running_corrects / samples_per_phase[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_epoch_num = epoch
                        save_path = os.path.join(BEST_MODLE_PATH, f'Folder{fold}_{FILE_NAME}_best_model.pth')
                        torch.save(model.state_dict(), save_path)
                        print(f'Model improved and saved to {save_path}')

        print(f'\nHighest Val Acc: {best_acc:.4f} in Epoch {best_epoch_num}\n')
        
        # Test
        custom_order = ['T1', 'T2', 'FLAIR', 'DWI', 'DTI', 'OTHER']
        class_names = test_dataset.classes
        best_model_path = os.path.join(BEST_MODLE_PATH, f'Folder{fold}_{FILE_NAME}_best_model.pth')
        model.load_state_dict(torch.load(best_model_path))

        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                labels_mapped = [custom_order.index(class_names[label]) for label in labels.cpu().numpy()]
                preds_mapped = [custom_order.index(class_names[pred]) for pred in preds.cpu().numpy()]
                
                all_preds.extend(preds_mapped)
                all_labels.extend(labels_mapped)

        # if args.model == 'googlenet':
        #     pickle_save_path = os.path.join(BEST_MODLE_PATH, f'Folder{fold}_{FILE_NAME}_test_labels_and_preds.pkl')
        #     with open(pickle_save_path, 'wb') as f:
        #         pickle.dump({'all_labels': all_labels, 'all_preds': all_preds}, f)

        accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        print(f'Accuracy on test set: {100 * accuracy:.2f}%')
        results.append(float(accuracy))

    mean_value = statistics.mean(results)
    std_dev = statistics.stdev(results)

    print('\n', results)
    print("Mean:", mean_value)
    print("Standard Deviation:", std_dev)