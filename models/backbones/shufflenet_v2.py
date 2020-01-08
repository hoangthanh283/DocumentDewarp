import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.backbone_base import BackBoneBase


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x

def conv_1x1_bn(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True))
        
def conv_bn(in_c, out_c, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True))


class ShuffleBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2
        if downsample:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )
            
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                
                nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )
        else:
            assert in_c == out_c
            self.branch2 = nn.Sequential(
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),

                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )

    def forward(self, x):
        out = None
        if self.downsample:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = torch.cat((x1, self.branch2(x2)), 1)
        return channel_shuffle(out, 2)
    

class ShuffleNet2(nn.Module):
    def __init__(self, input_channel=1, output_channel=224, net_type=1):
        super(ShuffleNet2, self).__init__()
        self.stage_repeat_num = [4, 8, 4]
        if net_type == 0.5:
            self.out_channels = [input_channel, 24, 48, 96, 192, 1024]
        elif net_type == 1:
            self.out_channels = [input_channel, 24, 116, 232, 464, 1024]
        elif net_type == 1.5:
            self.out_channels = [input_channel, 24, 176, 352, 704, 1024]
        elif net_type == 2:
            self.out_channels = [input_channel, 24, 244, 488, 976, 2948]
        else:
            print("the type is error, you should choose 0.5, 1, 1.5 or 2")
        
        # building layers
        # self.conv1 = nn.Conv2d(input_channel, self.out_channels[1], 3, 2, 1)
        self.conv1 = nn.Conv2d(input_channel, self.out_channels[1], 3, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_c = self.out_channels[1]
        
        self.stages = []
        for stage_idx in range(len(self.stage_repeat_num)):
            out_c = self.out_channels[2 + stage_idx]
            repeat_num = self.stage_repeat_num[stage_idx]
            for i in range(repeat_num):
                if i == 0:
                    self.stages.append(ShuffleBlock(in_c, out_c, downsample=True))
                else:
                    self.stages.append(ShuffleBlock(in_c, in_c, downsample=False))
                in_c = out_c

        self.stages = nn.Sequential(*self.stages)
        in_c = self.out_channels[-2]
        out_c = self.out_channels[-1]
        self.conv5 = conv_1x1_bn(in_c, output_channel, 1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.stages(x)
        x = self.conv5(x)
        return x


class ShuffleNetV2Encoder(BackBoneBase):
    """ An backbone for image embedding
    """
    def __init__(self, input_channel, output_channel, net_type):
        super(ShuffleNetV2Encoder, self).__init__()
        self.encoder = ShuffleNet2(input_channel, output_channel, net_type)
    
    @classmethod
    def load_opt(cls, opt):
        return cls(
            input_channel=opt.input_channel,
            output_channel=opt.output_channel,
            net_type=1.5
        )
    
    def forward(self, x):
        "See :obj:`.backbones.backbone_base.BackBoneBase.forward()`"
        return self.encoder(x)  

if __name__ == "__main__":
    model = ShuffleNetV2Encoder(input_channel=1, output_channel=256, net_type=0.5)
    out = model(torch.rand([4, 1, 200, 200]))
    print(out.shape)