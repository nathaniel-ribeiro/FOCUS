import tensorflow_datasets as tfds
import tensorflow as tf
import os
os.environ["TFDS_TEMP_DIR"] = "gs://thanny/tfds_tmp/"
train_ds, val_ds, test_ds = tfds.load(name="i_naturalist2021", split=["mini", "val", "test"], data_dir="gs://thanny/")
