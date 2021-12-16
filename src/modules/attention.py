import torch
from torch import nn


class SelectiveAttention(nn.Module):
    """
    Simply bag_sent * relations (no learned parameters)
    """

    def __init__(self, device):
        super(SelectiveAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)  # sentence dimension
        self.device = device

    def masking(self, bag, bag_size):
        # mask padding elements
        tmp = torch.arange(bag.size(1)).repeat(bag.size(0), 1).unsqueeze(-1).to(self.device)
        mask = torch.lt(tmp, bag_size[:, None, None].repeqt(1, tmp.size(1), 1))
        return mask

    def forward(self, bags, bags_size, relation_embeds):
        mask = self.masking(bags, bags_size)
        scores = torch.matmul(bags, relation_embeds.transpose(0, 1))
        scores = torch.where(mask, scores, torch.full_like(scores, float('-inf')).to(self.device))
        scores = self.softmax(scores)

        sent_rep = torch.matmul(scores.permute(0, 2, 1), bags)
        return sent_rep
