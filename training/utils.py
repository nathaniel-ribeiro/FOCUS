import yaml
from dataclasses import dataclass
import os
import json
import torch

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    return device

@dataclass
class TrainingOptions:
    data_dir: str
    target_size: tuple
    batch_size: int
    model_name: str
    pretrained: str
    learning_rate: float
    epochs: int
    save_freq: int

def load_config_file(path):
    with open(path) as f:
        params = yaml.safe_load(f)
    
    params['target_size'] = tuple(params['target_size'])
    return TrainingOptions(**params)

def load_categories():
    options = load_config_file('config.yaml')
    data_dir = options.data_dir
    categories_filepath = os.path.join(data_dir, 'categories.json')
    with open(categories_filepath) as f:
        categories = json.load(f)
    return categories

def load_micronet_metadata():
    options = load_config_file('config.yaml')
    data_dir = options.data_dir
    metadata_filepath = os.path.join(data_dir, 'micronet_annotations.json')
    with open(metadata_filepath) as f:
        metadata = json.load(f)
    return metadata
