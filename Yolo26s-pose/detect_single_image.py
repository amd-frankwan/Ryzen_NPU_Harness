import cv2
import numpy as np
import onnxruntime as ort

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image to a square while preserving aspect ratio."""
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
    return im, r, (dw, dh)

def run_onnx_pose(onnx_model_path, image_path, conf_threshold=0.25):
    # 1. Load ONNX model session
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 2. Read and Preprocess Image
    orig_img = cv2.imread(image_path)
    h_orig, w_orig, _ = orig_img.shape
    
    # Letterbox to 640x640 (standard YOLO size)
    input_size = 640
    img, ratio, (dw, dh) = letterbox(orig_img, (input_size, input_size))
    
    # BGR to RGB, normalize to [0, 1], and change layout to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)  
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 3, 640, 640)

    # 3. Run Inference
    outputs = session.run(None, {input_name: img})
    predictions = np.squeeze(outputs[0])  # Shape typically: (num_predictions, 56)

    # 4. Post-processing & Drawing
    # Layout of 56 values: [box_x1, box_y1, box_x2, box_y2, score, kpt1_x, kpt1_y, kpt1_conf, ...]
    for pred in predictions:
        score = pred[4]
        if score < conf_threshold:
            continue
            
        # Rescale bounding box back to original image coordinates
        x1 = int((pred[0] - dw) / ratio)
        y1 = int((pred[1] - dh) / ratio)
        x2 = int((pred[2] - dw) / ratio)
        y2 = int((pred[3] - dh) / ratio)
        
        # Draw bounding box
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, f"Person: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Extract and rescale 17 keypoints
        keypoints = pred[5:].reshape(-1, 3) # Shape: (17, 3) -> [x, y, conf]
        for kpt_idx, kpt in enumerate(keypoints):
            kpt_x, kpt_y, kpt_conf = kpt
            if kpt_conf > 0.5:  # Keypoint confidence threshold
                # Rescale keypoint coordinates
                real_x = int((kpt_x - dw) / ratio)
                real_y = int((kpt_y - dh) / ratio)
                
                # Draw joint point
                cv2.circle(orig_img, (real_x, real_y), 5, (0, 0, 255), -1)

    # Save and show window
    cv2.imwrite("output_pose.jpg", orig_img)
    cv2.imshow("YOLO26s-Pose ONNX (No Ultralytics)", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = "yolo26s-pose.onnx"
    IMAGE_PATH = "path/to/your/image.jpg"
    run_onnx_pose(MODEL_PATH, IMAGE_PATH)