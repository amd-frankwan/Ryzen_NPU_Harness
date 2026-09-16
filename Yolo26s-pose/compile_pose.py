import onnxruntime
import os

Model = "yolo26s-pose_int8_headfp"
model_name = os.getcwd() + "/" + Model + ".onnx"


provider_options_dict1 = {
    "config_file": "vaiml_config.json",
    "cache_dir": "./",
    "cache_key": Model,
    "target": "VAIML"
}

session = onnxruntime.InferenceSession(
    model_name,
    providers=["VitisAIExecutionProvider"],
    provider_options=[provider_options_dict1]
)

print("Providers:", session.get_providers())
print("Compilation completed")
