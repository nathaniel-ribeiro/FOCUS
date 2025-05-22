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

options = load_config_file('config.yaml')
data_dir = options.data_dir
categories = load_categories()

imagenet_templates = [
    'a bad photo of a {label}.',
    'a photo of many {label}.',
    'a sculpture of a {label}.',
    'a photo of the hard to see {label}.',
    'a low resolution photo of the {label}.',
    'a rendering of a {label}.',
    'graffiti of a {label}.',
    'a bad photo of the {label}.',
    'a cropped photo of the {label}.',
    'a tattoo of a {label}.',
    'the embroidered {label}.',
    'a photo of a hard to see {label}.',
    'a bright photo of a {label}.',
    'a photo of a clean {label}.',
    'a photo of a dirty {label}.',
    'a dark photo of the {label}.',
    'a drawing of a {label}.',
    'a photo of my {label}.',
    'the plastic {label}.',
    'a photo of the cool {label}.',
    'a close-up photo of a {label}.',
    'a black and white photo of the {label}.',
    'a painting of the {label}.',
    'a painting of a {label}.',
    'a pixelated photo of the {label}.',
    'a sculpture of the {label}.',
    'a bright photo of the {label}.',
    'a cropped photo of a {label}.',
    'a plastic {label}.',
    'a photo of the dirty {label}.',
    'a jpeg corrupted photo of a {label}.',
    'a blurry photo of the {label}.',
    'a photo of the {label}.',
    'a good photo of the {label}.',
    'a rendering of the {label}.',
    'a {label} in a video game.',
    'a photo of one {label}.',
    'a doodle of a {label}.',
    'a close-up photo of the {label}.',
    'a photo of a {label}.',
    'the origami {label}.',
    'the {label} in a video game.',
    'a sketch of a {label}.',
    'a doodle of the {label}.',
    'a origami {label}.',
    'a low resolution photo of a {label}.',
    'the toy {label}.',
    'a rendition of the {label}.',
    'a photo of the clean {label}.',
    'a photo of a large {label}.',
    'a rendition of a {label}.',
    'a photo of a nice {label}.',
    'a photo of a weird {label}.',
    'a blurry photo of a {label}.',
    'a cartoon {label}.',
    'art of a {label}.',
    'a sketch of the {label}.',
    'a embroidered {label}.',
    'a pixelated photo of a {label}.',
    'itap of the {label}.',
    'a jpeg corrupted photo of the {label}.',
    'a good photo of a {label}.',
    'a plushie {label}.',
    'a photo of the nice {label}.',
    'a photo of the small {label}.',
    'a photo of the weird {label}.',
    'the cartoon {label}.',
    'art of the {label}.',
    'a drawing of the {label}.',
    'a photo of the large {label}.',
    'a black and white photo of a {label}.',
    'the plushie {label}.',
    'a dark photo of a {label}.',
    'itap of a {label}.',
    'graffiti of the {label}.',
    'a toy {label}.',
    'itap of my {label}.',
    'a photo of a cool {label}.',
    'a photo of a small {label}.',
    'a tattoo of the {label}.',
]

class INaturalistDataset(data.Dataset):
    def __init__(self, root_directory, annotations_filepath, augmentations, prompt_generation_mode="deterministic"):
        with open(annotations_filepath) as f:
            annotations_data = json.load(f)
        
        self.root_directory = root_directory
        self.image_filenames = [annotation['file_name'] for annotation in annotations_data['images']]
        
        # image filepath is root_directory/kingdom/species_id/filename so index 2 is what we want
        self.ids = [int(Path(image_filename).parts[2]) for image_filename in self.image_filenames]
        
        self.augmentations = augmentations
        if prompt_generation_mode not in ["deterministic", "stochastic"]:
            raise ValueError("Acceptable values for prompt_generation_mode are deterministic or stochastic")
        self.prompt_generation_mode = prompt_generation_mode
    
    def get_prompt(taxonomy):
        if self.prompt_generation_mode == "deterministic":
            return f"A photo of a {taxonomy.name}."
        else:
            random_template = random.choice(imagenet_templates)
            random_prompt = random_template.format(label=taxonomy.name)
            return random_prompt
        
    def __getitem__(self, idx):
        path = os.path.join(data_dir, self.image_filenames[idx])
        image_id = self.ids[idx]
        image_taxonomy = Taxonomy(*categories[image_id].values())
        image_prompt = get_prompt(image_taxonomy)
        image = Image.open(path).convert('RGB')
        image_numpy = np.array(image)
        augmented_image = self.augmentations(image=image_numpy)['image']
        return augmented_image, image_id, image_prompt
    
    def __len__(self):
        return len(self.image_filenames)
        