import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
import threading
import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO

app = FastAPI()
train_ds_numpy, val_ds_numpy, test_ds_numpy = None, None, None

with open('server_config.yaml') as file:
    params = yaml.safe_load(file)

data_dir = params['data_dir']
port = params['port']

@app.on_event("startup")
def load_dataset():
    global train_ds_numpy, val_ds_numpy, test_ds_numpy
    train_ds, val_ds, test_ds = tfds.load(name="i_naturalist2021", split=["train", "val", "test"], data_dir=data_dir, as_supervised=True)
    train_ds_numpy, val_ds_numpy, test_ds_numpy = train_ds.as_numpy(), val_ds.as_numpy(), test_ds.as_numpy()

#TODO: compress these 3 functions into a single one that takes the index and the name of the partition (["train", "val", "test"])
#TODO: add a get function that takes the name of the partition to get dataset partition size

@app.get("/train/{idx}")
def get_training_sample_by_index(idx: int):
    #TODO: find a better way to get the ith example of the dataset without iterating up to it
    for i, example in enumerate(train_ds_numpy):
        if i == idx:
            print(example)

@app.get("/val/{idx}")
def get_validation_sample_by_index(idx: int):
    #TODO: find a better way to get the ith example of the dataset without iterating up to it
    for i, example in enumerate(val_ds_numpy):
        if i == idx:
            print(example)

@app.get("/test/{idx}")
def get_test_sample_by_index(idx: int):
    #TODO: find a better way to get the ith example of the dataset without iterating up to it
    for i, example in enumerate(test_ds_numpy):
        if i == idx:
            print(example)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=port)