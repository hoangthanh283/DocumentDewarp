import torch
import logging
from torch import nn
from models.backbones import str2enc
from models.modules.coord_conv import (
    CoordConv, CoordConvNet, AddCoordinates
)
from models.modules.refinement_stages import (
    InitialStage, RefinementStage
)
from models.modules.head_stacks import (
    CompositeField, Shell, HeadStacks
)
from models.modules.side_pooling import (
   CornerPooling, CrossPooling, CenterPooling, \
   TopPool, BottomPool, LeftPool, RightPool
)



class KeyPointNet(nn.Module):
    """ Key point estimation network """
    def __init__(self, opt):
        super(KeyPointNet, self).__init__()
        self.opt = opt

        # Define coordinate convolution & add up 2 additional coordinate channels
        self.add_coordinates = AddCoordinates(with_r=False)
        self.opt.input_channel += 2
        self.encoder = str2enc[self.opt.backbone].load_opt(self.opt)

        ## Define center pooling
        #self.center_pooling = CenterPooling(self.opt.output_channel)
        
        # For addition scale, margin, width, spread, keypoint headers
        self.headers = CompositeField(in_features=self.opt.output_channel)
        self.net = Shell(self.encoder, [self.headers], net_scale=8)
        self.initialize_weights(self.net)
        self.net.process_heads = HeadStacks(
            [(v * 3 + 1, v * 3 + 2) for v in range(10)])

        # For multi refinement stages 
        self.initial_stage = InitialStage(\
            self.opt.output_channel, self.opt.num_heatmaps, self.opt.num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(self.opt.num_refinement_stages):
            self.refinement_stages.append(\
                RefinementStage(self.opt.output_channel + self.opt.num_heatmaps + self.opt.num_pafs, \
                    self.opt.output_channel, self.opt.num_heatmaps, self.opt.num_pafs))
    
    def initialize_weights(self, net):
        for m in net.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # avoid numerical instabilities
                # (only seen sometimes when training with GPU)
                # Variances in pretrained models can be as low as 1e-17.
                # m.running_var.clamp_(min=1e-8)
                m.eps = 1e-4  # tf default is 0.001

                # less momentum for variance and expectation
                m.momentum = 0.01  # tf default is 0.99

    def count_parameters(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'{num_params:,}'

    def forward(self, x):
        x = self.add_coordinates(x)
        backbone_features = self.encoder(x)
        #backbone_features = self.center_pooling(backbone_features)
        stages_output = self.initial_stage(backbone_features)

        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat(\
                    [backbone_features, stages_output[-2], \
                        stages_output[-1]], dim=1)))
        # heads_output = self.net(x)
        return stages_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='shufllenetv2')
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=256)
    parser.add_argument('--num_refinement_stages', type=int, default=1)
    parser.add_argument('--num_heatmaps', type=int, default=5)
    parser.add_argument('--num_pafs', type=int, default=8)
    opt = parser.parse_known_args()[0]

    model = KeyPointNet(opt)
    out = model(torch.rand([4, 1, 200, 200]))
    print(model.count_parameters())
