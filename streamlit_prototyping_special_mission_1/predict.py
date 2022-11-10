import models
import pytorch_lightning as pl
import torch
import albumentations
from PIL import Image
import io
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import streamlit as st

class MaskModel(pl.LightningModule):

    def __init__(self, model_name, num_class, learning_rate, loss_funtion_name):
        super().__init__()
        self.model = getattr(models, model_name)(num_class)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self.model(batch)
        pred = pred.argmax(dim=-1)
        return pred

@st.cache
def load_model(model_path,hparam_path):
    model = MaskModel.load_from_checkpoint(model_path,hparams_file= hparam_path).model
    model.eval()
    return model

def transform_image(image_bytes: bytes) -> torch.Tensor:
    transform = albumentations.Compose([
            albumentations.Resize(height=512, width=384),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.2, 0.2, 0.2)),
            ToTensorV2()
        ])
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image_array = np.array(image)
    return transform(image=image_array)['image'].unsqueeze(0)

def predict_image(model,image):
    image=transform_image(image)
    pred = model(image)
    pred = pred.argmax(dim=-1)
    return pred
