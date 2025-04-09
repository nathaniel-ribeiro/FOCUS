import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
import threading
import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO

app = FastAPI()
train_ds, val_ds, test_ds = None, None, None

with open('server_config.yaml') as file:
    params = yaml.safe_load(file)

data_dir = params['data_dir']
port = params['port']

@app.on_event("startup")
def load_dataset():
    global train_ds, val_ds, test_ds
    train_ds, val_ds, test_ds = tfds.load(name="i_naturalist2021", split=["train", "val", "test"], data_dir=data_dir, as_supervised=True)

@app.get("/train/{idx}")
def get_training_sample_by_index(idx: int):
    pass

@app.get("/val/{idx}")
def get_validation_sample_by_index(idx: int):
    pass

@app.get("/test/{idx}")
def get_test_sample_by_index(idx: int):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=port)