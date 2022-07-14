from torch import nn

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation_layer=None, dilation=1, padding=True):
        super().__init__()
        if padding == True:
            self.padding = (kernel_size - 1) // 2 * dilation
        else:
            self.padding = padding
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, dilation=dilation, groups=groups, bias=False)
        activation = activation_layer
        if activation_layer:
            self.net = nn.Sequential(
                conv2d,
                nn.BatchNorm2d(out_channels),
                activation
            )
        elif activation_layer == None:
            self.net = nn.Sequential(
                conv2d,
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.net(x)