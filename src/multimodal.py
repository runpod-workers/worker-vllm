import base64
import io
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor
import requests

class MultiModalProcessor:
    def __init__(self, engine_config):
        if engine_config.get("image_input_shape") is None:
            raise ValueError("image_input_shape must be provided in the engine config")
        
        self.image_shape = tuple(map(int, engine_config.get("image_input_shape").split(",")[-2:]))
        self.image_feature_size = int(engine_config.get("image_feature_size"))
        self.image_processor = AutoProcessor.from_pretrained(engine_config.get("model"))
        
    def url_to_base64(self, url):
        return base64.b64encode(requests.get(url).content).decode("utf-8")

    def base64_to_pixel_values(self, base64_string):
        image = Image.open(io.BytesIO(base64.b64decode(base64_string))).convert("RGB").resize(self.image_shape)
        pixel_values = self.image_processor("<image>"*self.image_feature_size, image, return_tensors='pt').to(0, torch.float16)
        image_tensor=pixel_values["pixel_values"]
        return image_tensor

