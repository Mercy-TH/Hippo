import os
from net.unetplusplus import UnetPlusPLus
from config.config import *
from scripts.loss import *
import onnx
from onnxsim import simplify

mode_path = '../models/unetplusplus_new.pth'
export_onnx_path = '../models/unetplusplus_new.onnx'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UnetPlusPLus(num_classes=num_classes, filters=filters, deep_supervision=deep_supervision).to(device)
if os.path.exists(mode_path):
    model.load_state_dict(torch.load(mode_path, map_location=device))
    print("Successful load weight.")


dummpy_input = torch.rand(1, 3, 128, 128).to(device)
print(dummpy_input.dtype)
torch.onnx.export(
    model,
    dummpy_input.float(),
    export_onnx_path,
    input_names=['input'],
    output_names=['output'],
    export_params=True,
    verbose=True
)
onnx_model = onnx.load(export_onnx_path)
model_simple, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simple, export_onnx_path)
print('ok.')

