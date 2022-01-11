import numpy as np
import torch


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
        Apply dropout to input tensor.
        Parameters
        ----------
        input_tensor: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
        Returns
        -------
        output: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
        """
    if drop_prob == 0 or drop_prob == None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask


def create_mask(x, N, device=None):
    if isinstance(x, torch.Tensor):
        x = x.data
    mask = np.zeros((len(x), N))
    for i in range(len(x)):
        mask[i, x:[i]] = 1
    return torch.Tensor(mask).to(device)
