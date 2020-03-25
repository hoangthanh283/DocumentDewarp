#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai(lxt@pku.edu.cn)
# GA module is borrowed from CGNL paper directly
# Pytorch implementation of GALD-Net

import torch
import torch.nn as nn
import torch.nn.functional as F



class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication. """
    def __init__(self, inplanes, planes, use_scale=False, groups=8):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                                                  groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNL block uses 'SCALE'", \
                   'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNL block uses '{}' groups".format(self.groups), \
                   'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []

            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual
        return x

class LocalAttenModule(nn.Module):
    # TODO: adaptive change the size of kernel size (default is 3 but fail!)
    def __init__(self, inplane, kernel_size=1):
        super(LocalAttenModule, self).__init__()
        self.dconv1 = nn.Sequential(
            nn.Conv2d(inplane, inplane, \
                kernel_size=kernel_size, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(inplane, inplane, \
                kernel_size=kernel_size, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(inplane, inplane, \
                kernel_size=kernel_size, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        res1 = x
        res2 = x
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = F.upsample(x, size=(h, w), mode="bilinear", align_corners=True)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1

class GALDBlock(nn.Module):
    def __init__(self, inplane, plane):
        super(GALDBlock, self).__init__()
        """ Note down the spatial into 1/16
        """
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane,kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.long_relation = SpatialCGNL(inplane, plane)
        self.local_attention = LocalAttenModule(inplane)

    def forward(self, x):
        size = x.size()[2:]
        x = self.down(x)
        x = self.long_relation(x)
        # local attention
        x = F.upsample(x,size=size, mode="bilinear", align_corners=True)
        res = x
        x = self.local_attention(x)
        return x + res

class GALDHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes):
        super(GALDHead, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        
        self.a2block = GALDBlock(interplanes, interplanes//2)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, outplanes, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(outplanes))

    def forward(self, x):
        output = self.conva(x)
        output = self.a2block(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output



if __name__ == '__main__':
    input_fts = torch.Tensor(1, 64, 769, 501)
    model = GALDHead(64, 128, 64)
    output = model(input_fts)
    print(output.size())