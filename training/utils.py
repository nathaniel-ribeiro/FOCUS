import yaml
from dataclasses import dataclass

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

def load_config_file(path):
    with open(path) as f:
        params = yaml.safe_load(f)
    
    params['target_size'] = tuple(params['target_size'])
    return TrainingOptions(**params)