import torch
from init_net import BaseNet


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
        total_loss, graph_loss, reco_loss, tmp_rel_probs, cur_adj = self.graph_encoder.learn_iter_graphs(
            graph_features,
            batch['source'].size(0),
            batch['bag_size'],
            batch['rel'],
            self.calc_task_loss)

        if tmp_rel_probs is not None:
            rel_probs = tmp_rel_probs
        else:
            rel_probs = task_rel_probs

        assert torch.sum(torch.isnan(rel_probs)) == 0.0
        return task_loss + total_loss, graph_loss, rel_probs, kld, reco_loss, mu_
