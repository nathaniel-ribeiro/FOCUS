import torch
import torch.nn as nn
import torch.utils.data as data
import torch.multiprocessing as mp
import os
import albumentations as A
import yaml
from inat_dataloader import INaturalistDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: move to config file/command line arg
TARGET_SIZE = (224, 224)
batch_size = 256

# NOTE: normalization is omitted since OpenCLIP has its own preprocessor
train_transforms = A.Compose([A.RandomResizedCrop(size=TARGET_SIZE), A.ToTensorV2()])
val_transforms = A.Compose([A.CenterCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], pad_if_needed=True), A.ToTensorV2()])

with open('config.yaml') as f:
    params = yaml.safe_load(f)

data_dir = params['data_dir']
train_root_dir = os.path.join(data_dir, 'train_val2018/')
val_root_dir = os.path.join(data_dir, 'train_val2018/')
train_annotations_filepath = os.path.join(data_dir, 'train2018.json')
val_annotations_filepath = os.path.join(data_dir, 'val2018.json')

train_dataset = INaturalistDataset(train_root_dir, train_annotations_filepath, train_transforms)
val_dataset = INaturalistDataset(val_root_dir, val_annotations_filepath, val_transforms)

# PyTorch recommends at most 8 processes or dataloader will risk freezing up
num_workers = min(mp.cpu_count(), 8)
train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
val_loader = data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)

for image_tensors, ids, _ in train_loader:
#     print(image_tensors['image'].shape)
#     print(len(ids))
    break
