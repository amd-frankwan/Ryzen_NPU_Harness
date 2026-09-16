"""YOLO26s-Pose (E2E) INT8 quantization — recommended head-exclude example.

Fixes both the detection-confidence AND keypoint-confidence collapse to 0.

Root cause (same mechanism for both):
  The model.23 end-to-end / NMS-free head concatenates coordinates (bbox &
  keypoint x,y, range 0-640) together with confidences (0-1) in the same
  `Concat` nodes. INT8 `Concat` requires ONE shared scale across all inputs, so
  the coordinate magnitude (~640) forces a coarse scale (=8.0). The confidence
  branch is dragged to that scale, and round(conf / 8.0) = 0 for every conf<=1.0.
    - detection conf : Sigmoid -> ... -> Concat_6  (output0)
    - keypoint conf  : Sigmoid_1 -> Concat_4 -> Reshape_10 -> Concat_5
  Excluding only the final subgraph (Concat_5 -> Concat_6) restores detection
  conf but NOT keypoint conf, because Sigmoid_1 / Concat_4 upstream stay INT8
  at scale 8.0.

Fix:
  Keep the ENTIRE model.23 decode/assembly in float. We compute the exclusion
  list straight from the ONNX graph = every /model.23/ node that is NOT part of
  the cv2/cv3/cv4 detection convs (name contains "one2one_cv"). This keeps the
  33 head convs quantized (INT8 on NPU) while the cheap decode ops stay float.
  The board runs this decode head on CPU (fallback subgraphs) anyway.

Verified on both opset17 (i-PRO export) and opset20: after this exclusion,
detection conf AND keypoint conf are non-zero across all 300x17 outputs.
"""
import os

import cv2
import numpy as np
import onnx

from onnxruntime.quantization.calibrate import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

FLOAT_MODEL = "yolo26s-pose.onnx"
QUANT_MODEL = "yolo26s-pose_int8_headfp.onnx"
CALIB_DIR = "/host_mount/coco2017/calib100"


class YOLO26PoseDataReader(CalibrationDataReader):
    def __init__(self, image_dir):
        self.files = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(".jpg")
        ]
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        img = cv2.imread(self.files[self.idx])
        self.idx += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img / 255.0
        return {"images": img}

    def rewind(self):
        self.idx = 0

def build_quantizer() -> ModelQuantizer:
    def head_decode_nodes(onnx_path):
        """Every model.23 decode/assembly node (NOT the cv2/cv3/cv4 detection convs).

        Robust across export variants (opset17/20, simplify on/off): selected by
        name, not by fixed index. Covers Concat_4/5/6, Sigmoid, Sigmoid_1, TopK,
        GatherElements, Slice, Reshape, Transpose, ReduceMax, Mod, Cast, Expand, ...
        """
        g = onnx.load(onnx_path).graph
        return [
            n.name
            for n in g.node
            if n.name.startswith("/model.23/") and "one2one_cv" not in n.name
        ]
    quant_config = get_default_config("VINT8")
    quant_config.extra_options["Int32Bias"] = False
    quant_config.enable_npu_cnn = True
    quant_config.extra_options["DedicatedQDQPair"] = True
    quant_config.extra_options["QuantizeAllOpTypes"] = True
    quant_config.extra_options["ActivationSymmetric"] = True
    quant_config.extra_options["TmpDir"] = f"$PWD"
    # Reference YOLOX quantization logs keep this option true even with a reader.
    quant_config.extra_options["UseRandomData"] = True
    # --- the recommended exclusion -------------------------------------------------
    exclude = head_decode_nodes(FLOAT_MODEL)
    quant_config.nodes_to_exclude = exclude
    # Do NOT also set subgraphs_to_exclude = [(Concat_5, Concat_6)] — nodes_to_exclude
    # above already covers the whole decode head (detection + keypoint conf paths).
    print(f"[quantize] excluding {len(exclude)} model.23 decode nodes (head kept in float):")
    for n in exclude:
        print("   ", n)
    # ------------------------------------------------------------------------------
    return ModelQuantizer(Config(global_quant_config=quant_config))

reader = YOLO26PoseDataReader(CALIB_DIR)
quantizer = build_quantizer()
quantizer.quantize_model(
    model_input=FLOAT_MODEL,
    model_output=QUANT_MODEL,
    calibration_data_reader=reader,
)
print(f"Done -> {QUANT_MODEL}")
