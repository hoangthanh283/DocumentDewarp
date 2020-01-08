import torch
from torch import nn
from models.backbones import str2enc


def conv(in_channels, \
    out_channels, 
    kernel_size=3, 
    padding=1, 
    bn=True, 
    dilation=1, 
    stride=1, 
    relu=True, 
    bias=True):
    modules = [nn.Conv2d(in_channels, \
        out_channels, kernel_size, stride, \
        padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]

class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features

class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]

class KeyPointNet(nn.Module):
    """ Key point estimation network """
    def __init__(self, opt):
        super(KeyPointNet, self).__init__()

        self.opt = opt
        self.encoder = str2enc[self.opt.backbone].load_opt(self.opt)
        self.initial_stage = InitialStage(self.opt.output_channel, self.opt.num_heatmaps, self.opt.num_pafs)
        self.refinement_stages = nn.ModuleList()
        
        for idx in range(self.opt.num_refinement_stages):
            self.refinement_stages.append(\
                RefinementStage(self.opt.output_channel + self.opt.num_heatmaps + self.opt.num_pafs, \
                    self.opt.output_channel, self.opt.num_heatmaps, self.opt.num_pafs))

    def count_parameters(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'{num_params:,}'

    def forward(self, x):
        backbone_features = self.encoder(x)
        stages_output = self.initial_stage(backbone_features)

        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat(\
                    [backbone_features, stages_output[-2], \
                        stages_output[-1]], dim=1)))
        return stages_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='mobilenet')
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=256)

    parser.add_argument('--num_refinement_stages', type=int, default=1)
    parser.add_argument('--num_heatmaps', type=int, default=5)
    parser.add_argument('--num_pafs', type=int, default=8)
    opt = parser.parse_known_args()[0]
    model = KeyPointNet(opt)
    out = model(torch.rand([4, 1, 200, 200]))
    print(out[0].shape, model.count_parameters())
