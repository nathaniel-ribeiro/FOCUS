import yaml
from dataclasses import dataclass
from functools import cache

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
    initial_temp: float

@cache
def load_config_file(path):
    with open(path) as f:
        params = yaml.safe_load(f)
    
    params['target_size'] = tuple(params['target_size'])
    return TrainingOptions(**params)

@cache
def load_categories():
    options = load_config_file('config.yaml')
    data_dir = options.data_dir
    categories_filepath = os.path.join(data_dir, 'categories.json')
    with open(categories_filepath) as f:
        categories = json.load(f)
    return categories