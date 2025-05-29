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
from utils import load_config_file, load_categories
import torch.multiprocessing as mp
import albumentations as A

options = load_config_file('config.yaml')
data_dir = options.data_dir
categories = load_categories()

DATA_DIR = options.data_dir
TARGET_SIZE = options.target_size
BATCH_SIZE = options.batch_size

# if more than 8 workers are used, dataloader may freeze up
NUM_WORKERS = min(mp.cpu_count() - 1, 8)

# NOTE: OpenCLIP normalization is replicated, skip OpenCLIP's built in preprocessor
OPENCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENCLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

BRIGHTNESS, CONTRAST, SATURATION, HUE = 0.4, 0.4, 0.4, 0.25

train_transforms = A.Compose([
    A.RandomResizedCrop(size=TARGET_SIZE),
    A.ColorJitter(BRIGHTNESS, CONTRAST, SATURATION, HUE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD), 
    A.ToTensorV2(),
])

test_transforms = A.Compose([
     A.CenterCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], pad_if_needed=True), 
     A.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD), 
     A.ToTensorV2(),
])

train_root_dir = os.path.join(DATA_DIR, 'train_val2018/')
test_root_dir = os.path.join(DATA_DIR, 'train_val2018/')
train_annotations_filepath = os.path.join(DATA_DIR, 'train2018.json')
test_annotations_filepath = os.path.join(DATA_DIR, 'val2018.json')

imagenet_templates = [
    'A bad photo of a {label}.',
    'A photo of many {label}.',
    'A sculpture of a {label}.',
    'A photo of the hard to see {label}.',
    'A low resolution photo of the {label}.',
    'A rendering of a {label}.',
    'Graffiti of a {label}.',
    'A bad photo of the {label}.',
    'A cropped photo of the {label}.',
    'A tattoo of a {label}.',
    'The embroidered {label}.',
    'A photo of a hard to see {label}.',
    'A bright photo of a {label}.',
    'A photo of a clean {label}.',
    'A photo of a dirty {label}.',
    'A dark photo of the {label}.',
    'A drawing of a {label}.',
    'A photo of my {label}.',
    'The plastic {label}.',
    'A photo of the cool {label}.',
    'A close-up photo of a {label}.',
    'A black and white photo of the {label}.',
    'A painting of the {label}.',
    'A painting of a {label}.',
    'A pixelated photo of the {label}.',
    'A sculpture of the {label}.',
    'A bright photo of the {label}.',
    'A cropped photo of a {label}.',
    'A plastic {label}.',
    'A photo of the dirty {label}.',
    'A JPEG corrupted photo of a {label}.',
    'A blurry photo of the {label}.',
    'A photo of the {label}.',
    'A good photo of the {label}.',
    'A rendering of the {label}.',
    'A {label} in a video game.',
    'A photo of one {label}.',
    'A doodle of a {label}.',
    'A close-up photo of the {label}.',
    'A photo of a {label}.',
    'The origami {label}.',
    'The {label} in a video game.',
    'A sketch of a {label}.',
    'A doodle of the {label}.',
    'A origami {label}.',
    'A low resolution photo of a {label}.',
    'The toy {label}.',
    'A rendition of the {label}.',
    'A photo of the clean {label}.',
    'A photo of a large {label}.',
    'A rendition of a {label}.',
    'A photo of a nice {label}.',
    'A photo of a weird {label}.',
    'A blurry photo of a {label}.',
    'A cartoon {label}.',
    'Art of a {label}.',
    'A sketch of the {label}.',
    'A embroidered {label}.',
    'A pixelated photo of a {label}.',
    'ITAP of the {label}.',
    'A JPEG corrupted photo of the {label}.',
    'A good photo of a {label}.',
    'A plushie {label}.',
    'A photo of the nice {label}.',
    'A photo of the small {label}.',
    'A photo of the weird {label}.',
    'The cartoon {label}.',
    'Art of the {label}.',
    'A drawing of the {label}.',
    'A photo of the large {label}.',
    'A black and white photo of a {label}.',
    'The plushie {label}.',
    'A dark photo of a {label}.',
    'ITAP of a {label}.',
    'Graffiti of the {label}.',
    'A toy {label}.',
    'ITAP of my {label}.',
    'A photo of a cool {label}.',
    'A photo of a small {label}.',
    'A tattoo of the {label}.',
]

class INaturalistDataset(data.Dataset):
    def __init__(self, root_directory, annotations_filepath, augmentations, random_prompts=False):
        with open(annotations_filepath) as f:
            annotations_data = json.load(f)
        
        self.root_directory = root_directory
        self.image_filenames = [annotation['file_name'] for annotation in annotations_data['images']]
        
        # image filepath is root_directory/kingdom/species_id/filename so index 2 is what we want
        self.ids = [int(Path(image_filename).parts[2]) for image_filename in self.image_filenames]
        
        self.augmentations = augmentations
        self.random_prompts = random_prompts
    
    def get_prompt(self, taxonomy):
        if self.random_prompts:
            random_template = random.choice(imagenet_templates)
            random_prompt = random_template.format(label=taxonomy.name)
            return random_prompt
        else:
            return taxonomy.name
        
    def __getitem__(self, idx):
        path = os.path.join(data_dir, self.image_filenames[idx])
        image_id = self.ids[idx]
        image_taxonomy = Taxonomy(*categories[image_id].values())
        image_prompt = self.get_prompt(image_taxonomy)
        image = Image.open(path).convert('RGB')
        image_numpy = np.array(image)
        augmented_image = self.augmentations(image=image_numpy)['image']
        return augmented_image, image_id, image_prompt
    
    def __len__(self):
        return len(self.image_filenames)

def make_datasets():   
    train_dataset = INaturalistDataset(train_root_dir, train_annotations_filepath, train_transforms, True)
    test_dataset = INaturalistDataset(test_root_dir, test_annotations_filepath, test_transforms, False)
    return train_dataset, test_dataset

def make_dataloaders():
    train_dataset, test_dataset = make_datasets()
    train_loader = data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, test_loader

def get_labels():
    return [Taxonomy(*category.values()).name for category in categories]
