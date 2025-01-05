# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import argparse
import os
import re
import sys
import time

import numpy as np
from PIL import Image

import axengine as axe
from axengine import axclrt_provider_name, axengine_provider_name


def load_model(model_path: str | os.PathLike, selected_provider: str, selected_device_id: int = 0):
    if selected_provider == 'AUTO':
        # Use AUTO to let the pyengine choose the first available provider
        return axe.InferenceSession(model_path)

    providers = []
    if selected_provider == axclrt_provider_name:
        provider_options = {"device_id": selected_device_id}
        providers.append((axclrt_provider_name, provider_options))
    if selected_provider == axengine_provider_name:
        providers.append(axengine_provider_name)

    return axe.InferenceSession(model_path, providers=providers)


def preprocess_image(
        image_path: str | os.PathLike,
        middle_step_size: (int, int) = (256, 256),
        final_step_size: (int, int) = (224, 224)
):
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
    img = img.resize(middle_step_size)

    # Crop the center 224x224
    crop_x = (middle_step_size[0] - final_step_size[0]) // 2
    crop_y = (middle_step_size[1] - final_step_size[1]) // 2
    img = img.crop((crop_x, crop_y, crop_x + final_step_size[0], crop_y + final_step_size[1]))

    # Convert to numpy array and change dtype to int
    img_array = np.array(img).astype("uint8")
    # Transpose to (1, C, H, W)
    # img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def get_top_k_predictions(output: list[np.ndarray], k: int = 5):
    # Get top k predictions
    top_k_indices = np.argsort(output[0].flatten())[-k:][::-1]
    top_k_scores = output[0].flatten()[top_k_indices]
    return top_k_indices, top_k_scores


def main(model_path, image_path, middle_step_size, final_step_size, k, repeat_times, selected_provider,
         selected_device_id):
    # Load the model
    session = load_model(model_path, selected_provider, selected_device_id)

    # Preprocess the image
    input_tensor = preprocess_image(image_path, middle_step_size, final_step_size)

    # Get input name and run inference
    input_name = session.get_inputs()[0].name
    time_costs = []
    output = None
    for i in range(repeat_times):
        t1 = time.time()
        output = session.run(None, {input_name: input_tensor})
        t2 = time.time()
        time_costs.append((t2 - t1) * 1000)

    # Get top k predictions
    top_k_indices, top_k_scores = get_top_k_predictions(output, k)

    # Print the results
    print("  ------------------------------------------------------")
    print(f"  Top {k} Predictions:")
    for i in range(k):
        print(f"    Class Index: {top_k_indices[i]:>3}, Score: {top_k_scores[i]:.3f}")

    print("  ------------------------------------------------------")
    print(
        f"  min =   {min(time_costs):.3f} ms   max =   {max(time_costs):.3f} ms   avg =   {sum(time_costs) / len(time_costs):.3f} ms"
    )
    print("  ------------------------------------------------------")


def parse_size(size_str):
    pattern = r'^\s*\d+\s*,\s*\d+\s*$'
    if not re.match(pattern, size_str):
        raise argparse.ArgumentTypeError(R'params should looks like: "height,width", such as: "256,256"')

    height, width = map(int, size_str.split(','))
    return height, width


class ExampleParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"\nError: {message}")
        print("\nExample usage:")
        print("  python3 classification.py -m <model_file> -i <image_file>")
        print("  python3 classification.py -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg")
        print(
            f"  python3 classification.py -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -p {axengine_provider_name}")
        print(
            f"  python3 classification.py -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -p {axclrt_provider_name}")
        sys.exit(1)


if __name__ == "__main__":
    ap = ExampleParser()
    ap.add_argument('-m', '--model-path', type=str, help='model path', required=True)
    ap.add_argument('-i', '--image-path', type=str, help='image path', required=True)
    ap.add_argument(
        '-s',
        '--resize-size',
        type=parse_size,
        help=R'imagenet resize size: "height,width", such as: "256,256"',
        default='256,256',
    )
    ap.add_argument(
        '-c',
        '--crop-size',
        type=parse_size,
        help=R'imagenet crop size: "height,width", such as: "224,224"',
        default='224,224',
    )
    ap.add_argument(
        '-k',
        '--top-k',
        type=int,
        help='top k predictions',
        default=5
    )
    ap.add_argument('-r', '--repeat', type=int, help='repeat times', default=100)
    ap.add_argument(
        '-p',
        '--provider',
        type=str,
        choices=["AUTO", f"{axclrt_provider_name}", f"{axengine_provider_name}"],
        help=f'"AUTO", "{axclrt_provider_name}", "{axengine_provider_name}"',
        default='AUTO'
    )
    ap.add_argument(
        '-d',
        '--device-id',
        type=int,
        help=R'axclrt device index, depends on how many cards inserted',
        default=0
    )
    args = ap.parse_args()

    model_file = args.model_path
    image_file = args.image_path

    # check if the model and image exist
    assert os.path.exists(model_file), f"model file path {model_file} does not exist"
    assert os.path.exists(image_file), f"image file path {image_file} does not exist"

    resize_size = args.resize_size
    crop_size = args.crop_size
    top_k = args.top_k

    repeat = args.repeat

    provider = args.provider
    device_id = args.device_id

    main(model_file, image_file, resize_size, crop_size, top_k, repeat, provider, device_id)
