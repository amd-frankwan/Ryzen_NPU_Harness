"""Single-image YOLO26s-Pose ONNX inference (no Ultralytics).

Matches the end-to-end / NMS-free `model.23` head that
quantize_yolo26s_pose_recommended.py keeps in float: output0 is
(1, max_det, C) with boxes already decoded to xyxy in letterboxed pixel
space, e.g. (1, 300, 57) = [x1, y1, x2, y2, score, class, 17*(x, y, conf)].
A dense (1, C, num_anchors) export is also handled, with NMS.
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

# COCO-17 skeleton, used only for drawing.
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image to a square while preserving aspect ratio.

    Returns the padded image, the scale ratio, and the *integer* left/top
    padding actually applied (so un-padding is exact rather than sub-pixel off).
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into top/bottom, left/right
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def _channels_last(arr):
    """Squeeze the batch dim and orient the array as (num_detections, channels)."""
    arr = arr[0] if arr.ndim == 3 else arr
    # The detection axis is always the longer one (300 vs 57, or 8400 vs 56).
    return arr if arr.shape[0] >= arr.shape[1] else arr.T


def _decode(pred, conf_threshold):
    """Split raw model output into (boxes_xyxy, scores, keypoints) in 640-space."""
    pred = _channels_last(pred)
    n, c = pred.shape

    if n > 1000:
        # Dense export: [cx, cy, w, h, nc class scores, 17*(x, y, conf)], needs NMS.
        nc = c - 4 - 17 * 3
        if nc < 1:
            raise ValueError(f"unexpected dense output width {c}")
        scores = pred[:, 4:4 + nc].max(axis=1)
        keep = scores >= conf_threshold
        pred, scores = pred[keep], scores[keep]

        cxcy, wh = pred[:, :2], pred[:, 2:4]
        boxes = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=1)
        kpts = pred[:, 4 + nc:].reshape(-1, 17, 3)

        if len(boxes):
            idx = cv2.dnn.NMSBoxes(
                np.concatenate([boxes[:, :2], wh], axis=1).tolist(),
                scores.tolist(), conf_threshold, 0.45,
            )
            idx = np.asarray(idx).reshape(-1)
            boxes, scores, kpts = boxes[idx], scores[idx], kpts[idx]
        return boxes, scores, kpts

    # End-to-end head: [x1, y1, x2, y2, score, (class), 17*(x, y, conf)].
    # 56 columns -> no class column; 57 -> a class column sits at index 5.
    kpt_start = 5 if (c - 5) == 17 * 3 else 6
    if (c - kpt_start) != 17 * 3:
        raise ValueError(f"unexpected end-to-end output width {c}")

    scores = pred[:, 4]
    keep = scores >= conf_threshold
    pred, scores = pred[keep], scores[keep]
    return pred[:, :4], scores, pred[:, kpt_start:].reshape(-1, 17, 3)


def run_onnx_pose(onnx_model_path, image_path, conf_threshold=0.25,
                  kpt_threshold=0.5, output_path="output_pose.jpg", show=False,
                  use_npu=False, vitis_config="vaiml_config.json", cache_dir="./", cache_key=None):
    # 1. Load ONNX model session
    if use_npu:
        print(f"Using VitisAIExecutionProvider (NPU)")
        print(f"  Config: {vitis_config}")
        print(f"  Cache dir: {cache_dir}")
        providers = [
            (
                "VitisAIExecutionProvider",
                {
                    "config_file": str(vitis_config),
                    "target": "VAIML",
                    "cacheDir": str(cache_dir),
                    "cacheKey": cache_key,
                },
            )
        ]
    else:
        print("Using CPUExecutionProvider")
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_model_path, providers=providers)
    inp = session.get_inputs()[0]

    # Take the spatial size from the model rather than assuming 640.
    ih, iw = inp.shape[2], inp.shape[3]
    ih = 640 if not isinstance(ih, int) else ih
    iw = 640 if not isinstance(iw, int) else iw

    # 2. Read and Preprocess Image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    h_orig, w_orig = orig_img.shape[:2]

    img, ratio, (pad_x, pad_y) = letterbox(orig_img, (ih, iw))

    # BGR to RGB, normalize to [0, 1], and change layout to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 3, H, W)

    # 3. Run Inference
    outputs = session.run(None, {inp.name: img})
    boxes, scores, keypoints = _decode(outputs[0], conf_threshold)

    # 4. Un-letterbox everything at once, then clip to the original frame.
    boxes = (boxes - np.array([pad_x, pad_y, pad_x, pad_y])) / ratio
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w_orig - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h_orig - 1)
    keypoints[..., 0] = (keypoints[..., 0] - pad_x) / ratio
    keypoints[..., 1] = (keypoints[..., 1] - pad_y) / ratio

    # 5. Draw
    for box, score, kpts in zip(boxes, scores, keypoints):
        x1, y1, x2, y2 = box.round().astype(int)
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, f"Person: {score:.2f}", (x1, max(y1 - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        visible = kpts[:, 2] > kpt_threshold
        for a, b in SKELETON:
            if visible[a] and visible[b]:
                cv2.line(orig_img, tuple(kpts[a, :2].round().astype(int)),
                         tuple(kpts[b, :2].round().astype(int)), (255, 128, 0), 2)
        for (kx, ky, _), ok in zip(kpts, visible):
            if ok:
                cv2.circle(orig_img, (int(round(kx)), int(round(ky))), 4, (0, 0, 255), -1)

    print(f"{len(boxes)} detection(s) above conf {conf_threshold}")
    cv2.imwrite(output_path, orig_img)
    print(f"wrote {output_path}")

    if show:  # off by default: WSL sessions usually have no display
        cv2.imshow("YOLO26s-Pose ONNX (No Ultralytics)", orig_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="YOLO26s-Pose ONNX single-image inference")
    p.add_argument("image")
    p.add_argument("--model", default="yolo26s-pose.onnx")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--kpt-conf", type=float, default=0.5)
    p.add_argument("--out", default="output_pose.jpg")
    p.add_argument("--show", action="store_true")
    p.add_argument("--use-npu", action="store_true", help="Use VitisAIExecutionProvider (NPU)")
    p.add_argument("--vitis-config", type=str, default="vaiml_config.json", help="VitisAI config file path")
    p.add_argument("--cache-dir", type=str, default="./", help="Cache directory for NPU")
    p.add_argument("--cache-key", type=str, default=None, help="Cache key for NPU")
    a = p.parse_args()
    run_onnx_pose(a.model, a.image, a.conf, a.kpt_conf, a.out, a.show, a.use_npu, a.vitis_config, a.cache_dir, a.cache_key)
