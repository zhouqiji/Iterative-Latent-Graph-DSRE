import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGraph(nn.Module):
    def __init__(self, config, w_emb):
        super(TextGraph, self).__init__()
