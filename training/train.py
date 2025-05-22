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
from train import train

options = load_config_file('config.yaml')
DATA_DIR = options.data_dir
TARGET_SIZE = options.target_size
BATCH_SIZE = options.batch_size
NUM_WORKERS = options.num_workers
MAX_NUM_WORKERS = 8

if NUM_WORKERS > MAX_NUM_WORKERS:
    raise ValueError(f"Num workers should not exceed {MAX_NUM_WORKERS}")

# NOTE: normalization is omitted since OpenCLIP has its own preprocessor
train_transforms = A.Compose([A.RandomResizedCrop(size=TARGET_SIZE), A.ToTensorV2()])
val_transforms = A.Compose([A.CenterCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], pad_if_needed=True), A.ToTensorV2()])

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

trainer = Trainer(train_loader, val_loader, options)
trainer.train()