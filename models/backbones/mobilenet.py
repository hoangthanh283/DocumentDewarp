import torch
from torch import nn
from models.backbones.backbone_base import BackBoneBase


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )

class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class MobileNet(nn.Module):
    def __init__(self, input_channel=1, output_channel=128):
        super().__init__()
        self.input_dim = input_channel
        self.model = nn.Sequential(
            conv(self.input_dim, 32, stride=2, bias=False),
            conv_dw(32,  64),
            conv_dw(64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512), # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512) # conv5_5
        )
        self.cpm = Cpm(512, output_channel)

    def forward(self, x):
        assert self.input_dim == x.size(1), \
            "Input channel should be equal to {0}".format(self.input_dim)
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)
        return backbone_features


class MobileNetEncoder(BackBoneBase):
    """ An backbone for image embedding
    """
    def __init__(self, input_channel, output_channel):
        super(MobileNetEncoder, self).__init__()
        self.encoder = MobileNet(input_channel, output_channel)
    
    @classmethod
    def load_opt(cls, opt):
        return cls(
            input_channel=opt.input_channel,
            output_channel=opt.output_channel
        )
    
    def forward(self, x):
        "See :obj:`.backbones.backbone_base.BackBoneBase.forward()`"
        return self.encoder(x)  

if __name__ == "__main__":
    model = MobileNetEncoder(input_channel=1, output_channel=256)
    out = model(torch.rand([4, 1, 200, 200]))
    print(out.shape)