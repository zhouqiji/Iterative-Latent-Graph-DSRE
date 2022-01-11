import torch
from .constants import VERY_SMALL_NUMBER

def batch_normalize_adj(mx, mask=None):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    # mx: shape: [batch_size, N, N]

    # strategy 1)
    # rowsum = mx.sum(1)
    # r_inv_sqrt = torch.pow(rowsum, -0.5)
    # r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0. # I got this error: copy_if failed to synchronize: device-side assert triggered

    # strategy 2)
    rowsum = torch.clamp(mx.sum(1), min=VERY_SMALL_NUMBER)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    if mask is not None:
        r_inv_sqrt = r_inv_sqrt * mask

    r_mat_inv_sqrt = []
    for i in range(r_inv_sqrt.size(0)):
        r_mat_inv_sqrt.append(torch.diag(r_inv_sqrt[i]))

    r_mat_inv_sqrt = torch.stack(r_mat_inv_sqrt, 0)
    return torch.matmul(torch.matmul(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)
