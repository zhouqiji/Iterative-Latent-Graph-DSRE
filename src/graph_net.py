import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from init_net import BaseNet
import numpy as np
from modules.constants import *

from text_graph import TextGraph


class GraphNet(BaseNet):
    """
    The main model
    """

    # ----------------------------------------------- #

    def calc_task_loss(self, rel_logits, target):
        task_loss = self.task_loss(rel_logits, target.type_as(rel_logits))  # (examples, categories)
        task_loss = torch.sum(task_loss, dim=1)
        task_loss = torch.mean(task_loss)  # sum across relations, mean over batch
        rel_probs = torch.sigmoid(rel_logits)

        # Get the score

        return rel_probs, task_loss

    # model forward
    def forward(self, batch):
        ######################
        # Encoder
        ######################
        x_vec = self.w_embed(batch['source'])

        if self.config['include_positions']:
            pos1 = self.p_embed(batch['pos1'])
            pos2 = self.p_embed(batch['pos2'])
            x_vec = torch.cat([x_vec, pos1, pos2], dim=2)

        x_vec = self.in_drop(x_vec)

        ##########################
        # Graph Encoder
        ##########################
        graph_out, graph_features, reco_features = self.graph_encoder(x_vec, batch)

        kld, mu_ = reco_features

        task_rel_probs, task_loss = self.calc_task_loss(graph_out, batch['rel'])
        total_loss, graph_loss, reco_loss, tmp_rel_probs = self.graph_encoder.learn_iter_graphs(graph_features,
                                                                                                batch['source'].size(0),
                                                                                                batch['bag_size'],
                                                                                                batch['rel'],
                                                                                                self.calc_task_loss)

        # if self.config['constrain_loss']:
        #     l0 = out_adj.sum(2) / (sent_length.unsqueeze(1) + 1e-9)
        #     l0 = l0.sum(1) / (sent_length + 1e-9)
        #     l0 = l0.sum() / sent_length.size(0)
        #     # `l0` now has the expected selection rate for this mini-batch
        #     # we now follow the steps Algorithm 1 (page 7) of this paper:
        #     # https://arxiv.org/abs/1810.00597
        #     # to enforce the constraint that we want l0 to be not higher
        #     # than `self.selection` (the target sparsity rate)
        #
        #     # lagrange dissatisfaction, batch average of the constraint
        #     c0_hat = (l0 - self.config['constrain_rate'])
        #
        #     # moving average of the constraint
        #     self.c0_ma = self.lagrange_alpha * self.c0_ma + \
        #             (1 - self.lagrange_alpha) * c0_hat.item()
        #
        #     # compute smoothed constraint (equals moving average c0_ma)
        #     c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())
        #
        #     # update lambda
        #     self.lambda0 = self.lambda0 * torch.exp(
        #         self.lagrange_lr * c0.detach())
        #     graph_loss += self.lambda0.detach() * c0

        if tmp_rel_probs is not None:
            rel_probs = tmp_rel_probs
        else:
            rel_probs = task_rel_probs

        assert torch.sum(torch.isnan(rel_probs)) == 0.0
        return task_loss + total_loss, graph_loss, rel_probs, kld, reco_loss, mu_
