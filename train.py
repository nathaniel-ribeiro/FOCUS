import tensorflow_datasets as tfds
import tensorflow as tf

train_ds, val_ds, test_ds = tfds.load(name="i_naturalist2021", split=["train", "val", "test"], data_dir="gs://thanny/")
