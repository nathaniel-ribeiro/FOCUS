import tensorflow_datasets as tfds
import tensorflow as tf
import os
import albumentations as A
import yaml

with open('config.yaml') as f:
    params = yaml.safe_load(f)

data_dir = params['data_dir']

train_ds, val_ds, test_ds = tfds.load(name="i_naturalist2018", split=["train", "validation", "test"], download=True, data_dir=data_dir)