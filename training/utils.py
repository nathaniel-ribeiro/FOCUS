import yaml
from dataclasses import dataclass

@dataclass
class TrainingOptions:
    data_dir: str
    target_size: tuple
    batch_size: int
    num_workers: int

def load_config_file(path):
    with open(path) as f:
        params = yaml.safe_load(f)
    return TrainingOptions(**params)