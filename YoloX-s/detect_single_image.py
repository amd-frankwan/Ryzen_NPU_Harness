#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Run YOLOX ONNX object detection on one image and save an annotated output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont


DEFAULT_IMAGE = Path(__file__).resolve().with_name("example_coco.jpg")


COCO_CLASSES = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to YOLOX ONNX model.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path.")
    parser.add_argument("--output", type=Path, required=True, help="Annotated output image path.")
    parser.add_argument("--img-size", type=int, default=None, help="Model input size. Inferred when static.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Minimum object score.")
    parser.add_argument("--nms-thr", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--provider", choices=["npu", "cpu"], default="npu", help="Execution provider.")
    parser.add_argument("--config-file", type=Path, default=Path("vitisai_config.json"), help="VitisAI config file.")
    parser.add_argument("--cache-dir", type=Path, default=Path("."), help="VitisAI EP cache directory.")
    parser.add_argument("--cache-key", default=None, help="VitisAI EP cache key. Defaults to the model stem.")
    parser.add_argument("--input-format", choices=["bgr", "rgb"], default="bgr", help="Channel order sent to the model.")
    parser.add_argument("--skip-decode", action="store_true", help="Use when the ONNX output is already decoded.")
    return parser.parse_args()


def create_session(args: argparse.Namespace) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 1

    if args.provider == "cpu":
        print(f"Creating CPU inference session for {args.model} ...")
        return ort.InferenceSession(
            str(args.model),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    provider_options = {
        "config_file": str(args.config_file),
        "target": "VAIML",
        "cacheDir": str(args.cache_dir),
        "cacheKey": args.cache_key or args.model.stem,
    }
    print(f"Creating NPU inference session with VitisAIExecutionProvider for {args.model} ...")
    print(f"  cacheDir: {provider_options['cacheDir']}")
    print(f"  cacheKey: {provider_options['cacheKey']}")
    return ort.InferenceSession(
        str(args.model),
        sess_options=session_options,
        providers=["VitisAIExecutionProvider"],
        provider_options=[provider_options],
    )


def get_input_size(session: ort.InferenceSession, override: int | None) -> int:
    if override is not None:
        return override
    shape = session.get_inputs()[0].shape
    height, width = shape[-2], shape[-1]
    if isinstance(height, int) and isinstance(width, int) and height == width:
        return height
    raise ValueError("Could not infer static square input size. Pass --img-size.")


def preprocess(image: Image.Image, image_size: int, input_format: str) -> tuple[np.ndarray, float]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(image_size / height, image_size / width)
    resized_size = (int(width * scale), int(height * scale))
    resized = rgb.resize(resized_size, Image.Resampling.BILINEAR)

    padded = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    padded.paste(resized, (0, 0))
    array = np.asarray(padded, dtype=np.float32)
    if input_format == "bgr":
        array = array[:, :, ::-1]
    array = array.transpose(2, 0, 1)
    return np.expand_dims(np.ascontiguousarray(array), axis=0), scale


def normalize_predictions(output: np.ndarray) -> np.ndarray:
    predictions = np.asarray(output)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError(f"Expected 2D YOLOX predictions, got shape {predictions.shape}")
    if predictions.shape[0] in (84, 85) and predictions.shape[1] > predictions.shape[0]:
        predictions = predictions.T
    if predictions.shape[1] < 6:
        raise ValueError(f"Expected xywh + objectness + class scores, got shape {predictions.shape}")
    return predictions


def decode_yolox_raw(predictions: np.ndarray, image_size: int) -> np.ndarray:
    """Decode raw YOLOX grid outputs to absolute xywh coordinates."""
    decoded = predictions.copy()
    grids = []
    strides = []
    for stride in (8, 16, 32):
        height = image_size // stride
        width = image_size // stride
        yv, xv = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2).astype(np.float32))
        strides.append(np.full((height * width, 1), stride, dtype=np.float32))

    grid = np.concatenate(grids, axis=0)
    stride_values = np.concatenate(strides, axis=0)
    if decoded.shape[0] != grid.shape[0]:
        raise ValueError(f"Raw YOLOX decode expected {grid.shape[0]} boxes, got {decoded.shape[0]}")

    decoded[:, :2] = (decoded[:, :2] + grid) * stride_values
    decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * stride_values
    return decoded


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h
        union = areas[current] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-6)
        order = rest[iou <= threshold]

    return keep


def postprocess(
    output: np.ndarray,
    scale: float,
    image_size: tuple[int, int],
    input_size: int,
    score_thr: float,
    nms_thr: float,
    decode_raw: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = normalize_predictions(output)
    if decode_raw:
        predictions = decode_yolox_raw(predictions, input_size)

    boxes = xywh_to_xyxy(predictions[:, :4]) / scale
    objectness = predictions[:, 4]
    class_scores = predictions[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]

    valid = scores >= score_thr
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]

    width, height = image_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)

    kept_indices: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        keep = nms(boxes[indices], scores[indices], nms_thr)
        kept_indices.extend(indices[keep].tolist())

    kept_indices = sorted(kept_indices, key=lambda idx: float(scores[idx]), reverse=True)
    return boxes[kept_indices], scores[kept_indices], class_ids[kept_indices]


def draw_detections(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()

    for box, score, class_id in zip(boxes, scores, class_ids, strict=True):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        color = tuple(int(value) for value in np.random.default_rng(int(class_id)).integers(64, 256, size=3))
        label = COCO_CLASSES[int(class_id)] if int(class_id) < len(COCO_CLASSES) else f"class_{int(class_id)}"
        text = f"{label} {score:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_box = draw.textbbox((x1, y1), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_height - 4)
        draw.rectangle((x1, text_y, x1 + text_width + 4, text_y + text_height + 4), fill=color)
        draw.text((x1 + 2, text_y + 2), text, fill=(0, 0, 0), font=font)

    return output


def main() -> None:
    args = parse_args()
    session = create_session(args)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    image_size = get_input_size(session, args.img_size)
    print(f"  Input : {input_name}")
    print(f"  Outputs: {output_names}")
    print(f"  Image size: {image_size}")

    image = Image.open(args.image)
    input_tensor, scale = preprocess(image, image_size, args.input_format)
    outputs = session.run(output_names, {input_name: input_tensor})

    boxes, scores, class_ids = postprocess(
        outputs[0],
        scale,
        image.size,
        image_size,
        args.score_thr,
        args.nms_thr,
        decode_raw=not args.skip_decode,
    )
    annotated = draw_detections(image, boxes, scores, class_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.output)
    print(f"Wrote {args.output} with {len(boxes)} detections")


if __name__ == "__main__":
    main()