import torch
import torch.nn as nn
import torch.utils.data as data
import torch.multiprocessing as mp
import os
import albumentations as A
import yaml
from inat_dataloader import INaturalistDataset
from torch.utils.data._utils.collate import default_collate
from utils import load_config_file
from trainer import train

options = load_config_file('config.yaml')
DATA_DIR = options.data_dir
TARGET_SIZE = options.target_size
BATCH_SIZE = options.batch_size

# if more than 8 workers are used, dataloader may freeze up
NUM_WORKERS = min(mp.cpu_count() - 1, 8)

# NOTE: OpenCLIP normalization is replicated, skip OpenCLIP's built in preprocessor
OPENCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENCLIP_STD = [0.26862954, 0.26130258, 0.27577711]

train_transforms = A.Compose([
    A.RandomResizedCrop(size=TARGET_SIZE), 
    A.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD), 
    A.ToTensorV2(),
])

val_transforms = A.Compose([
     A.CenterCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], pad_if_needed=True), 
     A.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD), 
     A.ToTensorV2(),
])

train_root_dir = os.path.join(DATA_DIR, 'train_val2018/')
val_root_dir = os.path.join(DATA_DIR, 'train_val2018/')
train_annotations_filepath = os.path.join(DATA_DIR, 'train2018.json')
val_annotations_filepath = os.path.join(DATA_DIR, 'val2018.json')

train_dataset = INaturalistDataset(train_root_dir, train_annotations_filepath, train_transforms)
val_dataset = INaturalistDataset(val_root_dir, val_annotations_filepath, val_transforms)

def _collate(batch):
    images, ids, taxonomies = zip(*batch)
    images = default_collate(images)
    ids = torch.tensor(ids, dtype=torch.long)
    return images, ids, list(taxonomies)

train_loader = data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=_collate)
val_loader = data.DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=_collate)

train(train_loader)