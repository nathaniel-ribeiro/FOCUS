import yaml

def load_config_file(path):
    with open(path) as f:
        return yaml.safe_load(f)