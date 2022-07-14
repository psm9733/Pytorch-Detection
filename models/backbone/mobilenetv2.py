import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from utils.utils_pytorch import model_save_onnx
from torch import nn
from torchvision.models import mobilenet_v2

class mobilenetv2(nn.Module):
    def __init__(self, pretrained):
        self.backbone = None
        if pretrained:
            self.backbone = mobilenet_v2(pretrained=True)
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            self.backbone = mobilenet_v2(pretrained=False)
        
    def forward(self, x):
        out = self.backbone(x)
        return out


if __name__ == "__main__":
    model = mobilenet_v2(pretrained=True)
    model.eval()
    batch_size = 4
    input_shape = (3, 416, 416)
    dummy_input = torch.tensor(torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]))
    model_save_onnx(model, dummy_input, 'mobilenetv2')
    