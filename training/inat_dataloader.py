import torch
import yaml
from PIL import Image
import os
import json
import random
import numpy

with open('config.yaml') as f:
    params = yaml.safe_load(f)

data_dir = params['data_dir']

class INaturalistDataset(torch.utils.data.Dataset):
    def __init__(self, transforms):
        
        