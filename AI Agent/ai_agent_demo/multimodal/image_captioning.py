# -*- coding: utf-8 -*-
"""
@File    : image_captioning.py
@Time    : 2025/5/15 15:20
@Desc    : 
"""
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def caption_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


if __name__ == '__main__':
    result = caption_image("../data/Surfing_in_Hawaii.jpg")
    print(result)  # a man riding a wave on a surfboard
