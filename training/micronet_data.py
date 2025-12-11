import torch.utils.data as data
from PIL import Image
import os
import json
import random
import numpy
import yaml
from pathlib import Path
import numpy as np
from taxonomy import Taxonomy
from utils import load_config_file, load_micronet_metadata
import torch.multiprocessing as mp
import albumentations as A
import math
import pprint

options = load_config_file('config.yaml')
data_dir = options.data_dir
metadata = load_micronet_metadata()

DATA_DIR = options.data_dir
TARGET_SIZE = options.target_size
BATCH_SIZE = options.batch_size

# if more than 8 workers are used, dataloader may freeze up
NUM_WORKERS = min(mp.cpu_count() - 1, 8)

# NOTE: OpenCLIP normalization is replicated, skip OpenCLIP's built in preprocessor
OPENCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENCLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

train_transforms = A.Compose([
    A.RandomResizedCrop(size=TARGET_SIZE),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD), 
    A.ToTensorV2(),
])

test_transforms = A.Compose([
     A.CenterCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], pad_if_needed=True), 
     A.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD), 
     A.ToTensorV2(),
])

imagenet_templates = [
    'A bad photo of a {label}.',
    'A photo of many {label}.',
    'A photo of the hard to see {label}.',
    'A low resolution photo of the {label}.',
    'A bad photo of the {label}.',
    'A cropped photo of the {label}.',
    'A photo of a hard to see {label}.',
    'A bright photo of a {label}.',
    'A photo of a clean {label}.',
    'A photo of a dirty {label}.',
    'A dark photo of the {label}.',
    'A photo of my {label}.',
    'A photo of the cool {label}.',
    'A close-up photo of a {label}.',
    'A black and white photo of the {label}.',
    'A pixelated photo of the {label}.',
    'A bright photo of the {label}.',
    'A cropped photo of a {label}.',
    'A photo of the dirty {label}.',
    'A JPEG corrupted photo of a {label}.',
    'A blurry photo of the {label}.',
    'A photo of the {label}.',
    'A good photo of the {label}.',
    'A close-up photo of the {label}.',
    'A photo of a {label}.',
    'A low resolution photo of a {label}.',
    'A photo of a nice {label}.',
    'A photo of a weird {label}.',
    'A blurry photo of a {label}.',
    'A pixelated photo of a {label}.',
    'A JPEG corrupted photo of the {label}.',
    'A good photo of a {label}.',
    'A photo of the nice {label}.',
    'A photo of the small {label}.',
    'A photo of the weird {label}.',
    'A dark photo of a {label}.',
    'A photo of a cool {label}.',
    'A photo of a small {label}.',
]

def get_labels():
    labels = [item['species_guess'] for item in metadata.values()]
    return list(set(labels))

labels = get_labels()

class MicroNetDataset(data.Dataset):
    def __init__(self, ids_json_filepath, images_directory, augmentations, random_prompts=False):
        self.images_directory = images_directory
        self.augmentations = augmentations
        self.random_prompts = random_prompts
        with open(ids_json_filepath) as f:
            self.ids = json.load(f)

    def get_prompt(self, species_guess):
        if self.random_prompts:
            random_template = random.choice(imagenet_templates)
            random_prompt = random_template.format(label=species_guess)
            return random_prompt
        else:
            return species_guess
    
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        path = os.path.join(data_dir, self.images_directory, f'{image_id}.jpg')
        image_metadata = metadata[image_id]
        species_guess = image_metadata['species_guess']
        image_prompt = self.get_prompt(species_guess)
        image = Image.open(path).convert('RGB')
        image_numpy = np.array(image)
        augmented_image = self.augmentations(image=image_numpy)['image']

        return augmented_image, labels.index(species_guess), image_prompt
    
    def __len__(self):
        return len(self.ids)

def make_datasets():   
    train_dataset = MicroNetDataset(os.path.join(data_dir, "train.json"), "micronet_images", train_transforms, True)
    val_dataset = MicroNetDataset(os.path.join(data_dir, "val.json"), "micronet_images", test_transforms, False)
    test_dataset = MicroNetDataset(os.path.join(data_dir, "test.json"), "micronet_images", test_transforms, False)
    return train_dataset, val_dataset, test_dataset

def make_dataloaders():
    train_dataset, val_dataset, test_dataset = make_datasets()
    train_loader = data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = data.DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader