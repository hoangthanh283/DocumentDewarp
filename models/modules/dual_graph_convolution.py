import torch
import torch.nn as nn
import torch.nn.functional as F



class GCN(nn.Module):
    """ Graph convolution unit (single layer). """
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class DualGraphConv(nn.Module):
    """
    Dual graph convolutional network.
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, normalize=True):
        super(DualGraphConv, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        """ Graph convolution in coordinate space """
        # Coordinate space projection
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_theta = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_gamma = nn.Conv2d(num_in, self.num_n, kernel_size=1)
        self.conv_v = nn.Conv2d(num_in, self.num_n, kernel_size=1) 
        
        # Coordinate graph convolution & reprojection
        self.graph_conv_coord = GCN(self.num_s, self.num_n)
        self.conv_ws = nn.Conv1d(self.num_s, self.num_s, kernel_size=1)

        # Reprojection
        self.block_coord = nn.Conv2d(self.num_s, num_in, kernel_size=1)
        
        """ Graph convolution in feature space """
        # Feature space projection
        self.conv_ft_theta = nn.Sequential(
            nn.Conv2d(num_in, self.num_s, kernel_size=1),
            nn.BatchNorm2d(self.num_s),
            nn.ReLU()
        )
        self.conv_ft_phi = nn.Sequential(
            nn.Conv2d(num_in, self.num_n, kernel_size=1),
            nn.BatchNorm2d(self.num_n),
            nn.ReLU()
        )

        # Feature graph convolution
        self.graph_conv_space = GCN(self.num_s, self.num_n)

        # Reprojection
        self.block_space = nn.Sequential(
            nn.Conv2d(self.num_n, num_in, kernel_size=1),
            nn.BatchNorm2d(num_in),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        :param x: (n, d, h, w)
        '''
        n = x.size(0)
        x_origin = x

        """ Graph convolution in coordinate space """
        x = self.avg_pool(x)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_theta = self.conv_theta(x).view(n, self.num_s, -1)
        x_gamma = self.conv_gamma(x).view(n, self.num_n, -1)
        x_v = self.conv_v(x).view(n, self.num_n, -1)
        x_edges = torch.matmul(x_gamma.permute(0, 2, 1), x_v)
        x_edges = F.softmax(x_edges, dim=-1)

        # x_graph = self.graph_conv_coord(torch.matmul(x_theta, x_edges))
        x_graph = torch.matmul(x_theta, x_edges)
        x_graph = self.conv_ws(x_graph)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        self.upsampling = nn.UpsamplingNearest2d(size=(x_origin.shape[-2:]))
        x_reprojected_coord = self.upsampling(x_graph.view(n, self.num_s, *x.size()[2:]))
        x_coord_projection = self.block_coord(x_reprojected_coord)


        """ Graph convolution in feature space """
        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_ft_theta_prj = self.conv_ft_theta(x_origin).view(n, self.num_s, -1)
        x_ft_phi_prj = self.conv_ft_phi(x_origin).view(n, self.num_n, -1)

        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_ft_space = torch.matmul(x_ft_theta_prj, x_ft_phi_prj.permute(0, 2, 1))
        # x_ft_space = F.softmax(x_ft_space, dim=-1)
        
        if self.normalize:
            x_ft_space = x_ft_space * (1. / x_ft_phi_prj.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.graph_conv_space(x_ft_space)
        x_reprojected_space = torch.matmul(x_ft_theta_prj.permute(0, 2, 1), x_n_rel).permute(0, 2, 1)
        x_reprojected_space = x_reprojected_space.view(n, self.num_n, *x_origin.size()[2:])
        x_space_projection = self.block_space(x_reprojected_space)

        # final refined features
        out = x_origin + x_coord_projection + x_space_projection
        return out


if __name__ == '__main__':
    for normalize in [True, False]:
        data = torch.autograd.Variable(torch.randn(2, 32, 64, 120))
        net = DualGraphConv(32, 16, normalize=normalize)
        print(net(data).size())