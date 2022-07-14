import torch.onnx
import torch._C as _C
from torchsummary import summary
TrainingMode = _C._onnx.TrainingMode

def model_save_onnx(model, dummy_input, name, verbose = True):
    print("=================== Saving {} model ===================".format(name))
    if verbose:
        summary(model, dummy_input.shape[1:4], batch_size=dummy_input.shape[0], device='cpu')    
    torch.onnx.export(model, dummy_input, name + ".onnx", training=TrainingMode.TRAINING, opset_version=11)
