import tensorflow_datasets as tfds
import tensorflow as tf
import os
import albumentations as A

os.environ["TFDS_TEMP_DIR"] = "gs://thanny/tfds_tmp/"
train_ds, val_ds, test_ds = tfds.load(name="i_naturalist2021", split=["train", "val", "test"], data_dir="gs://thanny/", as_supervised=True)

train_ds_1 = train_ds.take(1)

for image, label in train_ds_1:
  print(image.shape)
  print(type(label))