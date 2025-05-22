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

with open('config.yaml') as f:
    params = yaml.safe_load(f)
data_dir = params['data_dir']

categories_filepath = os.path.join(data_dir, 'categories.json')
with open(categories_filepath) as f:
    categories = json.load(f)

class INaturalistDataset(data.Dataset):
    def __init__(self, root_directory, annotations_filepath, augmentations):
        with open(annotations_filepath) as f:
            annotations_data = json.load(f)
        
        self.root_directory = root_directory
        self.image_filenames = [annotation['file_name'] for annotation in annotations_data['images']]
        
        # image filepath is root_directory/kingdom/species_id/filename so index 2 is what we want
        self.ids = [int(Path(image_filename).parts[2]) for image_filename in self.image_filenames]
        
        self.augmentations = augmentations
        
    def __getitem__(self, idx):
        path = os.path.join(data_dir, self.image_filenames[idx])
        image_id = self.ids[idx]
        image_taxonomy = Taxonomy(*categories[image_id].values())
        image = Image.open(path).convert('RGB')
        image_numpy = np.array(image)
        augmented_image = self.augmentations(image=image_numpy)['image']
        return augmented_image, image_id, image_taxonomy
    
    def __len__(self):
        return len(self.image_filenames)
        