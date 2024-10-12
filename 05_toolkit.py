import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import argparse
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='')
parser.add_argument('--num_classes', type=int, default=6)
args = parser.parse_args()
print('\n\n', args, '\n\n')

class_names = ['DTI', 'DWI', 'FLAIR', 'OTHER', 'T1', 'T2']

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Data Path
DATA_PATH = args.path
BEST_MODLE_PATH = './02_models/best_model'

# Find the model ckpt
def find_specific_files(directory, file_keyword):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file_keyword in file:
                return os.path.join(root, file)

# Vote
def calculate_vote(row):
    most_common = row.mode()
    return most_common.iloc[0]

# Build dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                full_path = os.path.join(root, file)
                self.images.append(full_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        return image
    
    def get_path(self, idx):
        return self.images[idx]

transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')), 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    # Build data
    dataset = ImageDataset(root_dir=DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Get image path
    res = {}
    paths = []
    for i in range(len(dataset)):
        path = dataset.get_path(i)
        paths.append(path)
    res['Paths'] = paths

    # Get model path
    model_names = []
    model_paths = []
    for entry in os.listdir(BEST_MODLE_PATH):
        full_path = os.path.join(BEST_MODLE_PATH, entry)
        if os.path.isdir(full_path):
            model_names.append(entry)
            model_paths.append(full_path)

    specific_file_paths = []
    for path in model_paths:
        specific_files = find_specific_files(path, 'mid_best_model.pth')
        if specific_files:
            specific_file_paths.append(specific_files)

    print('model_names', model_names)
    print('model_paths', model_paths)
    print('specific_file_paths', specific_file_paths)

    # Set model and weight
    num_classes = args.num_classes

    for file_path in specific_file_paths:
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
            for images in tqdm(dataloader, desc="Processing images"):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                preds = preds.cpu().numpy()
                all_preds.extend(preds)

        res[model_name] = all_preds

    df = pd.DataFrame(res)

    df['vote'] = df.apply(calculate_vote, axis=1)
    mapping = {i: class_name for i, class_name in enumerate(class_names)}
    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: mapping.get(x, x))

    save_path = os.path.join('./', 'result.csv')
    df.to_csv(save_path, index=False)
