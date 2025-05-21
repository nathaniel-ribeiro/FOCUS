import torch
from PIL import Image
import os
import json
import random
import numpy
import yaml

with open('config.yaml') as f:
    params = yaml.safe_load(f)
data_dir = params['data_dir']

categories_filepath = os.path.join(data_dir, 'categories.json')
with open(categories_filepath) as f:
    categories = json.load(f)

class INaturalistDataset(torch.utils.data.Dataset):
    def __init__(self, root_directory, annotations_filepath, augmentations):
        with open(annotations_filepath) as f:
            annotations_data = json.load(f)
        self.image_filenames = [annotation['file_name'] for annotation in annotations_data]
        self.ids = [annotation['category_id'] for annotation in annotations_data]
        
    def __getitem__(self, idx):
        path = root_directory + self.image_filenames[idx]
        image_id = self.ids[idx]
        image_taxonomy = categories[image_id]
        image = Image.open(path).convert('RGB')
        augmented_image = augmentations(image)
        return augmented_image, image_id, image_taxonomy
    
    def __len__(self):
        return len(self.image_filenames)
        