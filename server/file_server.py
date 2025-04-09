import tensorflow as tf
import tensorflow_datasets as tfds
import yaml

with open('server_config.yaml') as file:
    params = yaml.safe_load(file)

def download_dataset():
    data_dir = params[data_dir]
    train_ds, val_ds, test_ds = tfds.load(name="i_naturalist2021", split=["train", "val", "test"], data_dir=data_dir, as_supervised=True)

if __name__ == "__main__":
    download_dataset()