from fastapi import FastAPI, File, UploadFile, Depends
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import io
from functools import lru_cache

app = FastAPI()

# Define models
class PredictResponse(BaseModel):
  message: str
  probability: float

# Define functions
def load_model(model_filepath, weights_filepath):
  json_file = open(model_filepath, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights(weights_filepath)
  return model

def img_preprocess(img):
  img = img.convert("RGB")
  img = img.resize((224, 224))
  img = np.array(img) / 255.0
  return np.expand_dims(img, axis=0)

# Get model
model_filepath = "resnet50.json"
weights_filepath = "resnet_ckpt3_epoch_04_val_loss_0.608.weights.h5"
model = load_model(model_filepath, weights_filepath)
# {'Normal': 0, 'bacteria': 1, 'virus': 2}
class_mapping = {0: "Bình thường", 1: "Viêm phổi do vi khuẩn", 2: "Viêm phổi do virus"}

# Define routes
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
  img = Image.open(io.BytesIO(await file.read()))
  img = img_preprocess(img)

  result = model.predict(img)
  class_index = np.argmax(result, axis=1)[0]
  return PredictResponse(
    message=class_mapping[class_index],
    probability=result[0][class_index]
  )