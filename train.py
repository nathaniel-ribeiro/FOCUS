import tensorflow_datasets as tfds
import tensorflow as tf
import os
import albumentations as A

train_ds, val_ds, test_ds = tfds.load(name="i_naturalist2021", split=["train", "val", "test"], as_supervised=True)