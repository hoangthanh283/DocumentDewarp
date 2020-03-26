import torch
import logging
from torch import nn
LOG = logging.getLogger(__name__)



class CompositeField(torch.nn.Module):
    dropout_p = 0.0
    quad = 0

    def __init__(self, head_name='pif', in_features=16, *,
                 n_fields=5,
                 n_confidences=1,
                 n_vectors=1,
                 n_scales=1,
                 kernel_size=1, padding=0, dilation=1):
        super(CompositeField, self).__init__()

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  head_name, n_fields, n_confidences, n_vectors, n_scales,
                  kernel_size, padding, dilation)

        self.shortname = head_name
        self.dilation = dilation

        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)
        self._quad = self.quad

        # classification
        out_features = n_fields * (4 ** self._quad)
        self.class_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_confidences)
        ])

        # regression
        self.reg_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, 2 * out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_vectors)
        ])
        self.reg_spreads = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in self.reg_convs
        ])

        # scale
        self.scale_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_scales)
        ])

        # dequad
        self.dequad_op = torch.nn.PixelShuffle(2)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        # classification
        classes_x = [class_conv(x) for class_conv in self.class_convs]
        if not self.training:
            classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

        # regressions
        regs_x = [reg_conv(x) * self.dilation for reg_conv in self.reg_convs]
        regs_x_spread = [reg_spread(x) for reg_spread in self.reg_spreads]
        regs_x_spread = [torch.nn.functional.leaky_relu(x + 2.0) - 2.0
                         for x in regs_x_spread]

        # scale
        scales_x = [scale_conv(x) for scale_conv in self.scale_convs]
        scales_x = [torch.nn.functional.relu(scale_x) for scale_x in scales_x]

        # upscale
        for _ in range(self._quad):
            classes_x = [self.dequad_op(class_x)[:, :, :-1, :-1]
                         for class_x in classes_x]
            regs_x = [self.dequad_op(reg_x)[:, :, :-1, :-1]
                      for reg_x in regs_x]
            regs_x_spread = [self.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                             for reg_x_spread in regs_x_spread]
            scales_x = [self.dequad_op(scale_x)[:, :, :-1, :-1]
                        for scale_x in scales_x]

        # reshape regressions
        regs_x = [
            reg_x.reshape(reg_x.shape[0],
                          reg_x.shape[1] // 2,
                          2,
                          reg_x.shape[2],
                          reg_x.shape[3])
            for reg_x in regs_x
        ]

        return classes_x + regs_x + regs_x_spread + scales_x

class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets, \
        net_scale=8, head_names=None, head_strides=None, \
        process_heads=None, cross_talk=0.0):
        super(Shell, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.head_names = head_names or [
            h.shortname for h in head_nets
        ]
        self.head_strides = head_strides or [
            net_scale // (2 ** getattr(h, '_quad', 0))
            for h in head_nets
        ]

        self.process_heads = process_heads
        self.cross_talk = cross_talk

    def forward(self, *args):
        image_batch = args[0]

        if self.training and self.cross_talk:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk
            
        x = self.base_net(image_batch)
        head_outputs = [hn(x) for hn in self.head_nets]

        if self.process_heads is not None:
            head_outputs = self.process_heads(*head_outputs)

        return head_outputs

class HeadStacks(torch.nn.Module):
    def __init__(self, stacks):
        super(HeadStacks, self).__init__()
        self.stacks_by_pos = {s[0]: s for s in stacks}
        self.ignore = {head_i for s in stacks for head_i in s[1:]}

    def forward(self, *args):
        heads = args

        stacked = []
        for head_i, head in enumerate(heads):
            if head_i in self.ignore:
                continue
            if head_i not in self.stacks_by_pos:
                stacked.append(head)
                continue

            fields = [heads[si] for si in self.stacks_by_pos[head_i]]
            stacked.append([
                torch.cat(fields_by_type, dim=1)
                for fields_by_type in zip(*fields)
            ])

        return stacked
