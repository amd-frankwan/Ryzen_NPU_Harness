# YOLOX-S

Object detection model: **YOLOX-S** (small) trained on **COCO** (80 classes).

| Property | Value |
| --- | --- |
| Source checkpoint | [yolox_s.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
| Input | `images` `[1, 3, 640, 640]` fp32, NCHW |
| Output | `output` `[1, 8400, 85]` fp32 |
| Opset | 17 |

---

### Host machine

Install the Python dependencies in your **host terminal**:

```bash
pip install torch==2.4.0 torchvision==0.19.0
pip install --no-build-isolation --no-deps yolox
# install dependencies of yolox
pip install onnx opencv-python-headless loguru scikit-image tqdm thop ninja tabulate tensorboard pycocotools
# install from requirements by removing the yolox
pip install -r ../../requirements.txt
pip install onnxscript
```


```bash
docker run -it --rm --network host \
    -v </yourLicenseDir>:/usr/licenses \
    -v <path>/<to>/cookbook:/cookbook \
    <Docker REPOSITORY:TAG> bash

# inside the container:
cd /cookbook/Models/YOLOX/yolox-s
```

The `onnxruntime` VitisAI Execution Provider, `../../../Utils/compile.py`, and Quark
are all provided by this Docker image.

Before quantizing, make the working directory writable (the container writes the quantized model, cache, and outputs here):

```bash
chmod -R a+w yolox-s
```

## 1. Export the model

Export the reference ONNX from official YOLOX checkpoint. Run this in
your **host terminal** or Vitis AI Docker as recommended (dependencies from Section 0):

```bash
python3 export_yolox_s.py
```

Outputs: `yolox_s.onnx`

---

## 2. Download COCO dataset (for calibration)

Download the COCO validation dataset for quantization calibration. Run this on
your **host machine**:

```bash
# Create dataset directory
mkdir -p coco && cd coco

# Download COCO 2017 validation images (~1GB, 5000 images)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Download COCO 2017 annotations (~241MB, required for accuracy evaluation)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

cd ..
```

Only a subset of images is needed for calibration (e.g., 10-100 images). The annotations are required for accuracy evaluation.

---

## 3. Quantize the model

Quantize to INT8 using Quark with COCO calibration data. Run this **inside the
Vitis AI Docker container** (see Section 0):

```bash
python3 quantize.py \
    --input yolox_s.onnx \
    --output yolox_s_int8.onnx \
    --calib-dir <path/to/coco/dataset> \
    --max-images 10
```

---

## 4. Compile the model

Run this **inside the Vitis AI Docker container** (see Section 0 — launch Docker
first):

```bash
python3 ../../../Utils/compile.py \
    --onnx_model_path yolox_s_int8.onnx \
    --config_file vitisai_config.json \
    --cache_dir ./ \
    --cache_key yolox_s_int8 \
    --enable-ai-analyzer
```

`../../../Utils/compile.py` runs the quantized model through the VitisAI Execution Provider (configured via `vitisai_config.json`, target VAIML). The `--enable-ai-analyzer` flag generates partition metadata required for runtime execution on the target device.

Emits the compiled artifact at `yolox_s_int8/yolox_s_int8.rai`.

---

## 5. Run inference (on board)

Run on the **target board**. First copy this repo onto the board and run the
board setup (see Section 0 — `scp` the dir, then run overlay setup):

```bash
ml_vart --app-config vart_config.json --benchmark --runs 100
```

`vart_config.json` points at the compiled `.rai` and feeds `input.bin` (FP32, BGR, unnormalized [0,255]).

---

## 6. Evaluate model accuracy

Evaluate INT8 model accuracy on COCO validation set.

> **Note:** By default, evaluates on all 5000 COCO validation images. Use `--max-images <N>` to limit the number of images evaluated.

### CPU evaluation (inside Vitis AI Docker)

```bash
python3 evaluate_coco.py \
    --onnx_model_path yolox_s_int8.onnx \
    --coco-dir <path/to/coco/dataset>
```

### NPU evaluation (on target board)

```bash
python3 evaluate_coco.py \
    --onnx_model_path yolox_s_int8.onnx \
    --coco-dir <path/to/coco/dataset> \
    --use-npu \
    --vitis-config vitisai_config.json \
    --cache-dir ./ \
    --cache-key yolox_s_int8
```

Outputs mAP, mAP@0.50, and mAP@0.75 metrics.

---
