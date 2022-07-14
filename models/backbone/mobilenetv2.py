import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from utils.utils_pytorch import model_save_onnx, show_features
from torch import nn
from torchvision.models import mobilenet_v2

class mobilenetv2(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.backbone = None
        if pretrained:
            self.backbone = mobilenet_v2(pretrained=True)
            # for p in self.backbone.parameters():
            #     p.requires_grad = False
        else:
            self.backbone = mobilenet_v2(pretrained=False)
        self.features = self.backbone.features
        self.stage_1 = nn.Sequential(
            self.features[:13],
            self.features[13].conv[:3],
        )
        
        self.stage_2 = nn.Sequential(
            self.features[14].conv[:3],
        )
        
    def forward(self, x):
        stage1_out = self.stage_1(x)
        stage2_out = self.stage_2(stage1_out)
        return [stage1_out, stage2_out]


if __name__ == "__main__":
    model = mobilenetv2(pretrained=True)
    model.eval()
    batch_size = 4
    input_shape = (3, 416, 416)
    dummy_input = torch.tensor(torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]).clone().detach())
    # show_features(model)
    model_save_onnx(model, dummy_input, 'mobilenetv2', True)
    