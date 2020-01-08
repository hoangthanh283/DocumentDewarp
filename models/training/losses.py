import torch 


def l2_loss(input, target, mask, batch_size):
    """L2 loss. Loss for a single two-dimensional vector. """
    loss = (input - target) * mask
    loss = (loss * loss) / 2 / batch_size
    return loss.sum()

def l1_loss(input, target, mask, batch_size, weight=None):
    """L1 loss. Loss for a single two-dimensional vector. """
    loss = torch.sqrt((input - target) ** 2) / 2 / batch_size
    if weight is not None:
        loss = loss * weight
    return loss.sum()

def l1_smooth_loss(input, target, mask, batch_size, weight=None, r_smooth=0.0, scale=1.0):
    """ L1 smooth loss with smooth threshold. """
    r = r_smooth * scale
    d = torch.sqrt((input - target)**2) / 2 / batch_size
    smooth_regime = d < r

    smooth_loss = 0.5 / r[smooth_regime] * d[smooth_regime] ** 2
    linear_loss = d[smooth_regime == 0] - (0.5 * r[smooth_regime == 0])
    losses = torch.cat((smooth_loss, linear_loss))

    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)

def laplace_loss(input, target, mask, logb, batch_size, weight=None):
    """Loss based on Laplace Distribution.
    Loss for a single two-dimensional vector with radial spread b.
    """

    # left derivative of sqrt at zero is not defined, so prefer torch.norm():
    # https://github.com/pytorch/pytorch/issues/2421
    # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    norm = (input - target).norm(dim=0)
    losses = 0.694 + logb + norm * torch.exp(-logb) / batch_size
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)