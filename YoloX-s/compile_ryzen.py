#Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

import onnxruntime

model='./yolox_s_int8.onnx'

#Compile for Ryzen
provider_options_dict = {
    "config_file": 'vaiml_config.json',
    "cache_dir":   './',
    "cache_key":   'yolox_s_int8',
    "target": "VAIML"

}
   
session = onnxruntime.InferenceSession(
    model,
    providers=["VitisAIExecutionProvider"],
    provider_options=[provider_options_dict]
)   
