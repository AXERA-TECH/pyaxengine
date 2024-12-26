# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#


import argparse
import cv2
import random
import colorsys
import axengine as axe
import numpy as np

CONF_THRESH = 0.45
IOU_THRESH = 0.45
STRIDES = [8, 16, 32]
ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]
NUM_OUTPUTS = 85
INPUT_SHAPE = [640, 640]


CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO_CATEGORIES = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90,
}


def letterbox_yolov5(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def pre_processing(image_raw, img_shape):

    img = letterbox_yolov5(image_raw, img_shape, stride=32, auto=False)[0]
    img = img[:, :, ::-1]
    img = img[np.newaxis, ...]
    origin_shape = image_raw.shape[0:2]
    return img, origin_shape


def draw_bbox(image, bboxes, classes=None, show_label=True, threshold=0.1):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    if classes == None:
        classes = {v: k for k, v in COCO_CATEGORIES.items()}

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors)
    )

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        if score < threshold:
            continue
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        print(
            f"bbox:{coor}, score:{score:5.4f}, label: {CLASS_NAMES[class_ind]:<10}, class: {class_ind}"
        )
        if show_label:
            bbox_mess = "%s: %.2f" % (CLASS_NAMES[class_ind], score)
            t_size = cv2.getTextSize(
                bbox_mess, 0, fontScale, thickness=bbox_thick // 2
            )[0]
            cv2.rectangle(
                image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1
            )

            cv2.putText(
                image,
                bbox_mess,
                (c1[0], c1[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                (0, 0, 0),
                bbox_thick // 2,
                lineType=cv2.LINE_AA,
            )

    return image


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def bboxes_iou(boxes1, boxes2):
    """calculate the Intersection Over Union value"""
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(proposals, iou_threshold, conf_threshold, multi_label=False):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    xc = proposals[..., 4] > conf_threshold
    proposals = proposals[xc]
    proposals[:, 5:] *= proposals[:, 4:5]
    bboxes = xywh2xyxy(proposals[:, :4])
    if multi_label:
        mask = proposals[:, 5:] > conf_threshold
        nonzero_indices = np.argwhere(mask)
        if nonzero_indices.size < 0:
            return
        i, j = nonzero_indices.T
        bboxes = np.hstack(
            (bboxes[i], proposals[i, j + 5][:, None], j[:, None].astype(float))
        )
    else:
        confidences = proposals[:, 5:]
        conf = confidences.max(axis=1, keepdims=True)
        j = confidences.argmax(axis=1)[:, None]

        new_x_parts = [bboxes, conf, j.astype(float)]
        bboxes = np.hstack(new_x_parts)

        mask = conf.reshape(-1) > conf_threshold
        bboxes = bboxes[mask]

    classes_in_img = list(set(bboxes[:, 5]))
    bboxes = bboxes[bboxes[:, 4].argsort()[::-1][:300]]
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = bboxes[:, 5] == cls
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1 :]]
            )
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.0
            cls_bboxes = cls_bboxes[score_mask]
    best_bboxes = np.vstack(best_bboxes)
    best_bboxes = best_bboxes[best_bboxes[:, 4].argsort()[::-1]]
    return best_bboxes


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.0)


def gen_proposals(outputs):
    new_pred = []
    anchor_grid = np.array(ANCHORS).reshape(-1, 1, 1, 3, 2)
    for i, pred in enumerate(outputs):
        pred = sigmoid(pred)
        n, h, w, c = pred.shape

        pred = pred.reshape(n, h, w, 3, 85)
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[..., 0:2]
        conv_raw_dwdh = pred[..., 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

        xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(np.float32)
        pred_xy = (conv_raw_dxdy * 2.0 - 0.5 + xy_grid) * STRIDES[i]
        pred_wh = (conv_raw_dwdh * 2) ** 2 * anchor_grid[i]
        pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

        new_pred.append(np.reshape(pred, (-1, np.shape(pred)[-1])))

    return np.concatenate(new_pred, axis=0)


def clip_coords(boxes, shape):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def post_processing(outputs, origin_shape, input_shape):

    proposals = gen_proposals(outputs)
    pred = nms(
        proposals, IOU_THRESH, CONF_THRESH, multi_label=True
    )  # set multi_label to true for testing map and then cost more time.
    pred[:, :4] = scale_coords(input_shape, pred[:, :4], origin_shape)
    return pred


def detect_yolov5(model_path, image_path, save_path, backend='auto', device_id=-1):

    if backend == 'auto':
        session = axe.InferenceSession(model_path, device_id)
    elif backend == 'ax':
        session = axe.AXInferenceSession(model_path)
    elif backend == 'axcl':
        session = axe.AXCLInferenceSession(model_path, device_id)
    image_data = cv2.imread(image_path)
    inputs, origin_shape = pre_processing(image_data, (640, 640))
    inputs = np.ascontiguousarray(inputs)
    results = session.run(None, {"images": inputs})
    det = post_processing(results, origin_shape, (640, 640))
    ret_image = draw_bbox(image_data, det)
    cv2.imwrite(save_path, ret_image)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="yolov5 example")
    parser.add_argument("--model", type=str, required=True, help="axmodel path")
    parser.add_argument("--image_path", type=str, required=True, help="image path")
    parser.add_argument('-b', '--backend', type=str, help='auto/ax/axcl', default='auto')
    parser.add_argument('-d', '--device_id', type=int, help='axcl device no, -1: onboard npu, >0: axcl devices', default=0)
    parser.add_argument(
        "--save_path", type=str, default="save.jpg", help="save image path"
    )
    args = parser.parse_args()
    assert args.backend in ['auto', 'ax', 'axcl'], "backend must be ax or axcl"
    assert args.device_id >= -1, "device_id must be greater than -1"
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"model             : {args.model}")
    print(f"image path        : {args.image_path}")
    print(f"backend           : {args.backend}")
    print(f"device_id         : {args.device_id}")
    print(f"save draw image to: {args.save_path}")
    detect_yolov5(args.model, args.image_path, args.save_path, args.backend, args.device_id)
# python3 yolov5_example.py --model /opt/data/npu/models/yolov5s.axmodel --image_path /opt/data/npu/images/dog.jpg --save_path ./detect_dog.jpg
