# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import axengine as axe
import numpy as np
from PIL import Image


def load_model(model_path):
    session = axe.InferenceSession(model_path)
    return session


def preprocess_image(image_path, target_size=(256, 256), crop_size=(224, 224)):
    # Load the image
    img = Image.open(image_path).convert("RGB")

    # Get original dimensions
    original_width, original_height = img.size

    # Determine the shorter side and calculate the center crop
    if original_width < original_height:
        crop_area = original_width
    else:
        crop_area = original_height

    crop_x = (original_width - crop_area) // 2
    crop_y = (original_height - crop_area) // 2

    # Crop the center square
    img = img.crop((crop_x, crop_y, crop_x + crop_area, crop_y + crop_area))

    # Resize the image to 256x256
    img = img.resize(target_size)

    # Crop the center 224x224
    crop_x = (target_size[0] - crop_size[0]) // 2
    crop_y = (target_size[1] - crop_size[1]) // 2
    img = img.crop((crop_x, crop_y, crop_x + crop_size[0], crop_y + crop_size[1]))

    # Convert to numpy array and change dtype to int
    img_array = np.array(img).astype("uint8")
    # Transpose to (1, C, H, W)
    # img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def get_top_k_predictions(output, k=5):
    # Get top k predictions
    top_k_indices = np.argsort(output[0].flatten())[-k:][::-1]
    top_k_scores = output[0].flatten()[top_k_indices]
    return top_k_indices, top_k_scores


def main(model_path, image_path, target_size, crop_size, k):
    # Load the model
    session = load_model(model_path)

    # Preprocess the image
    input_tensor = preprocess_image(image_path, target_size, crop_size)

    # Get input name and run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})

    # Get top k predictions
    top_k_indices, top_k_scores = get_top_k_predictions(output, k)

    # Print the results
    print(f"Top {k} Predictions:")
    for i in range(k):
        print(f"Class Index: {top_k_indices[i]}, Score: {top_k_scores[i]}")


if __name__ == "__main__":
    MODEL_PATH = "/opt/data/npu/models/mobilenetv2.axmodel"
    IMAGE_PATH = "/opt/data/npu/images/cat.jpg"
    TARGET_SIZE = (256, 256)  # Resize to 256x256
    CROP_SIZE = (224, 224)  # Crop to 224x224
    K = 5  # Top K predictions
    main(MODEL_PATH, IMAGE_PATH, TARGET_SIZE, CROP_SIZE, K)
