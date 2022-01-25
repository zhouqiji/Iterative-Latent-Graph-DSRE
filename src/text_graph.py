import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from modules.gvae import GVAE
from modules.gnn import GCN
from modules.graphlearn import GraphLearner, get_binarized_kneighbors_graph
from modules.utils import batch_normalize_adj

from modules.constants import VERY_SMALL_NUMBER

from helpers.common import *


class TextGraph(nn.Module):
    def __init__(self, config, lang_encoder=None, num_rel=0):
        super(TextGraph, self).__init__()
        self.config = config
        self.name = 'TextGraph'
        self.device = config['device']

        # Dropout
        self.dropout = config['graph_dropout']

        # Graph
        self.graph_learn = config['graph_learn']
        self.graph_metric_type = config['graph_metric_type']
        self.graph_module = config['graph_module']
        self.graph_skip_conn = config['graph_skip_conn']
        self.graph_include_self = config['graph_include_self']
        self.graph_batch_norm = config['graph_batch_norm']

        self.in_dim = config['word_embed_dim'] + 2 * config['pos_embed_dim']
        self.enc_dim = config['enc_dim']
        self.graph_hid_dim = config['graph_hid_dim']
        self.graph_out_dim = config['graph_out_dim']
        self.output_rel_dim = num_rel

        # G-VAE
        self.lat_dim = config['latent_dim']
        self.dec_dim = config['dec_dim']

        # Text Sentence Embedding
        self.ctx_encoder = lang_encoder

        self.linear_out = nn.Linear(self.graph_out_dim, self.output_rel_dim)
        self.hidden_out = nn.Linear(self.graph_hid_dim, self.graph_hid_dim)

        # Graph VAE
        self.gvae = GVAE(self.enc_dim, self.dec_dim, self.lat_dim, self.dropout)

        if self.graph_module == 'gcn':
            gcn_module = GCN
            self.encoder = gcn_module(nfeat=self.enc_dim,
                                      nhid=self.graph_hid_dim,
                                      nclass=self.graph_out_dim,
                                      graph_hops=config['graph_hops'],
                                      dropout=self.dropout,
                                      batch_norm=self.graph_batch_norm)
        else:
            raise RuntimeError('Unknown graph_module: {}'.format(self.graph_module))

        # Graph Learn
        if self.graph_learn:
            graph_learn_fun = GraphLearner
            self.graph_learner = graph_learn_fun(self.in_dim,
                                                 config['graph_learn_hidden_size'],
                                                 topk=config['graph_learn_topk'],
                                                 epsilon=config['graph_learn_epsilon'],
                                                 num_pers=config['graph_learn_num_pers'],
                                                 metric_type=config['graph_metric_type'],
                                                 device=self.device)
            self.graph_learner2 = graph_learn_fun(self.graph_hid_dim,
                                                  config['graph_learn_hidden_size2'],
                                                  topk=config['graph_learn_topk2'],
                                                  epsilon=config['graph_learn_epsilon2'],
                                                  num_pers=config['graph_learn_num_pers'],
                                                  metric_type=config['graph_metric_type'],
                                                  device=self.device)
        else:
            self.graph_learner = None
            self.graph_learner2 = None

    def compute_no_gnn_output(self, context, context_lens):
        context_vec = self.ctx_encoder(context, len_=context_lens)
        return context_vec

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, node_mask=None, graph_include_self=False,
                    init_adj=None):
        if self.graph_learn:
            raw_adj = graph_learner(node_features, node_mask)

            if self.graph_metric_type in ('kernel', 'weighted_cosine'):
                assert raw_adj.min().item() >= 0
                adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

            else:
                adj = torch.softmax(raw_adj, dim=-1)

            if graph_skip_conn in (0, None):
                if graph_include_self:
                    adj = adj + torch.eye(adj.size(0)).to(self.device)
            else:
                adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

            return raw_adj, adj
        else:
            raw_adj = None
            adj = init_adj
            return raw_adj, adj

    def compute_output(self, output_vec, bag_size, node_mask=None):

        output = self.graph_maxpool(output_vec.transpose(-1, -2))
        output = pad_sequence(torch.split(output, bag_size.tolist(), dim=0),
                              batch_first=True,
                              padding_value=0)
        # output = output.sum(-2)
        output = self.graph_maxpool(output.transpose(-1, -2))
        output = self.linear_out(output)
        return output

    def compute_hidden(self, output):
        out_hid = self.hidden_out(output)
        out_hid = torch.dropout(out_hid, self.dropout, self.training)
        out_hid = out_hid.sum(-2)
        return out_hid

    def mask_output(self, bag, bag_size):
        # mask padding elements
        tmp = torch.arange(bag.size(1)).repeat(bag.size(0), 1).unsqueeze(-1).to(self.device)
        mask = torch.lt(tmp, bag_size[:, None, None].repeat(1, tmp.size(1), 1))
        return mask

    def graph_maxpool(self, node_vec, node_mask=None):
        graph_embed = F.max_pool1d(node_vec, kernel_size=node_vec.size(-1)).squeeze(-1)
        return graph_embed

    def compute_init_adj(self, features, knn_size, mask=None):
        adj = get_binarized_kneighbors_graph(features, knn_size, mask=mask, device=self.device)
        adj_norm = batch_normalize_adj(adj, mask=mask)
        return adj_norm

    def prepare_init_graph(self, context, context_lens):
        context_mask = create_mask(context_lens, context.size(-2), device=self.device)
        raw_context_vec = context

        context_vec, (hidden, cell_state) = self.ctx_encoder(raw_context_vec, context_lens)

        init_adj = self.compute_init_adj(raw_context_vec.detach(), self.config['input_graph_knn_size'],
                                         mask=context_mask)
        return raw_context_vec, context_vec, context_mask, init_adj, hidden, cell_state

    def merge_tokens(self, enc_seq, mentions):
        """
        Merge tokens into mentions;
        Find which tokens belong to a mention (based on start-end ids) and average them
        """
        start1, end1, w_ids1 = torch.broadcast_tensors(mentions[:, 0].unsqueeze(-1),
                                                       mentions[:, 1].unsqueeze(-1),
                                                       torch.arange(0, enc_seq.shape[1]).unsqueeze(0).to(self.device))

        start2, end2, w_ids2 = torch.broadcast_tensors(mentions[:, 2].unsqueeze(-1),
                                                       mentions[:, 3].unsqueeze(-1),
                                                       torch.arange(0, enc_seq.shape[1]).unsqueeze(0).to(self.device))

        index_t1 = (torch.ge(w_ids1, start1) & torch.le(w_ids1, end1)).float().to(self.device).unsqueeze(1)
        index_t2 = (torch.ge(w_ids2, start2) & torch.le(w_ids2, end2)).float().to(self.device).unsqueeze(1)

        arg1 = torch.div(torch.matmul(index_t1, enc_seq), torch.sum(index_t1, dim=2).unsqueeze(-1)).squeeze(1)  # avg
        arg2 = torch.div(torch.matmul(index_t2, enc_seq), torch.sum(index_t2, dim=2).unsqueeze(-1)).squeeze(1)  # avg
        return arg1, arg2

    def graph_reco_loss(self, preds, labels, mu, log_var, n_nodes, norm, pos_weight):
        cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * log_var - mu.pow(2) - log_var.exp().pow(2), 1
        ))
        return cost, KLD

    def add_batch_graph_loss(self, out_adj, features, keep_batch_dim=False):
        # Graph regularization
        if keep_batch_dim:
            graph_loss = []
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss.append(self.config['smoothness_ratio'] * torch.trace(
                    torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(
                    np.prod(out_adj.shape[1:])))

            graph_loss = torch.Tensor(graph_loss).to(self.device)

            ones_vec = torch.ones(out_adj.shape[:-1], device=self.device)
            graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(
                torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).squeeze(-1).squeeze(-1) / \
                          out_adj.shape[-1]
            graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2), (1, 2)) / int(
                np.prod(out_adj.shape[1:]))

        else:
            graph_loss = 0
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss += self.config['smoothness_ratio'] * torch.trace(
                    torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape))

            ones_vec = torch.ones(out_adj.shape[:-1], device=self.device)
            graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(
                torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).sum() / out_adj.shape[0] / \
                          out_adj.shape[-1]
            graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))

        return graph_loss

    def SquaredFrobeniusNorm(self, X):
        return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))

    def batch_SquaredFrobeniusNorm(self, X):
        return torch.sum(torch.pow(X, 2), (1, 2)) / int(np.prod(X.shape[1:]))

    def batch_diff(self, X, Y, Z):
        assert X.shape == Y.shape
        diff_ = torch.sum(torch.pow(X - Y, 2), (1, 2))
        norm_ = torch.sum(torch.pow(Z, 2), (1, 2))
        diff_ = diff_ / torch.clamp(norm_, min=VERY_SMALL_NUMBER)
        return diff_

    def learn_iter_graphs(self, graph_features, source_size, bag_size, targets, loss_fn):
        init_adj, cur_raw_adj, cur_adj, raw_node_vec, init_node_vec, node_vec, node_mask = graph_features
        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            graph_loss = self.add_batch_graph_loss(cur_raw_adj, raw_node_vec)
        else:
            graph_loss = 0

        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        # Simper version
        if self.training:
            max_iter = self.config['graph_learn_max_iter']
        else:
            max_iter = self.config['graph_learn_max_iter'] * 2

        eps_adj = float(self.config['eps_adj'])

        # For graph learning
        loss = 0
        iter_ = 0

        # Indicate the last iteration umber for each example
        batch_last_iters = torch.zeros(source_size, dtype=torch.uint8, device=self.device)
        # Indicate either an xample is in ongoing state (i.e., 1) or stopping state (i.e., 0)
        batch_stop_indicators = torch.ones(source_size, dtype=torch.uint8, device=self.device)
        batch_all_outputs = []

        while self.config['graph_learn'] and (
                iter_ == 0 or torch.sum(batch_stop_indicators).item() > 0) and iter_ < max_iter:
            iter_ += 1
            batch_last_iters += batch_stop_indicators
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner2, node_vec,
                                                    self.graph_skip_conn,
                                                    node_mask=node_mask,
                                                    graph_include_self=self.graph_include_self,
                                                    init_adj=init_adj)

            update_adj_ratio = self.config['update_adj_ratio']
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj

            node_vec = torch.relu(self.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.config['gl_dropout'], training=self.training)

            # Add mid GNN layers if needed
            for encoder in self.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, self.config['gl_dropout'], training=self.training)

            tmp_output_sent = self.encoder.graph_encoders[-1](node_vec, cur_adj)

            # sentence representation
            tmp_output = self.compute_output(tmp_output_sent, bag_size)

            batch_all_outputs.append(tmp_output_sent.unsqueeze(1))

            _, tmp_loss = loss_fn(tmp_output, targets)
            if len(tmp_loss.shape) == 2:
                tmp_loss = torch.mean(tmp_loss, 1)

            loss += batch_stop_indicators.float() * tmp_loss

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                tmp_graph_loss = self.add_batch_graph_loss(cur_raw_adj, raw_node_vec,
                                                           keep_batch_dim=True)
                loss += batch_stop_indicators.float() * tmp_graph_loss

            if self.config['graph_learn'] and not self.config['graph_learn_ratio'] in (None, 0):
                loss += batch_stop_indicators.float() * self.batch_SquaredFrobeniusNorm(cur_adj - pre_raw_adj) * \
                        self.config['graph_learn_ratio']
            tmp_stop_criteria = self.batch_diff(cur_raw_adj, pre_raw_adj, first_raw_adj) > eps_adj
            batch_stop_indicators = batch_stop_indicators * tmp_stop_criteria

        if iter_ > 0:
            loss = torch.mean(loss / batch_last_iters.float()) + graph_loss

            batch_all_outputs = torch.cat(batch_all_outputs, 1)
            selected_iter_index = batch_last_iters.long().unsqueeze(-1) - 1
            if len(batch_all_outputs.shape) == 4:
                selected_iter_index = selected_iter_index.unsqueeze(-1).expand(-1, batch_all_outputs.size(-2),
                                                                               batch_all_outputs.size(-1)).unsqueeze(1)
                output = batch_all_outputs.gather(1, selected_iter_index).squeeze(1)
            else:
                output = batch_all_outputs.gather(1, selected_iter_index)

            output = self.compute_output(output, bag_size)
            rel_probs, _ = loss_fn(output, targets)
        else:
            loss = graph_loss
            rel_probs = None

        return loss, rel_probs

    def forward(self, context_vec, context_len, mentions, bag_size):
        # Prepare init node embedding, init adj
        raw_context_vec, context_vec, context_mask, init_adj, enc_hidden, cell_state = self.prepare_init_graph(
            context_vec, context_len)

        arg1, arg2 = self.merge_tokens(context_vec, mentions)  # contextualised representations of argument
        context_vec = context_vec + arg1.unsqueeze(dim=-2) + arg2.unsqueeze(dim=-2)

        # Init
        raw_node_vec = raw_context_vec  # word embedding
        init_node_vec = context_vec  # hidden embedding
        node_mask = context_mask

        cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner, raw_node_vec, self.graph_skip_conn,
                                                node_mask=node_mask, graph_include_self=self.graph_include_self,
                                                init_adj=init_adj)

        # node_vec = torch.relu(self.encoder.graph_encoders[0](init_node_vec, cur_adj))
        # node_vec = F.dropout(node_vec, self.dropout, training=self.training)
        #
        # # Add mid GNN layers
        # for encoder in self.encoder.graph_encoders[1:-1]:
        #     node_vec = torch.relu(encoder(node_vec, cur_adj))
        #     node_vec = F.dropout(node_vec, self.dropout, training=self.training)
        #
        # # Graph Output
        # output = self.encoder.graph_encoders[-1](node_vec, cur_adj)

        if self.config['reconstruction']:
            # TODO: Graph VAE mu_ var_ to get output
            mean_adj_sum = cur_adj.sum() / cur_adj.size(0)
            pos_weight = float(cur_adj.size(-1) * cur_adj.size(-1) - mean_adj_sum) / mean_adj_sum
            norm = cur_adj.size(-1) * cur_adj.size(-1) / float(cur_adj.size(-1) * cur_adj.size(-1) - mean_adj_sum * 2)
            adj_label = cur_adj
            # Reconstruction Graph
            reco_adj, mu_, logvar_ = self.gvae(init_node_vec, cur_adj, node_mask)
            reco_loss, kld = self.graph_reco_loss(reco_adj, adj_label, mu=mu_, log_var=logvar_,
                                                  n_nodes=adj_label.size(-1),
                                                  norm=norm, pos_weight=pos_weight.detach())
            node_vec = torch.relu(self.encoder.graph_encoders[0](init_node_vec, reco_adj))
            node_vec = F.dropout(node_vec, self.dropout, training=self.training)

            # Add mid GNN layers
            for encoder in self.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, reco_adj))
                node_vec = F.dropout(node_vec, self.dropout, training=self.training)

            # Graph Output
            output = self.encoder.graph_encoders[-1](node_vec, reco_adj)

            output = self.compute_output(output, bag_size)

        else:
            node_vec = torch.relu(self.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.dropout, training=self.training)

            # Add mid GNN layers
            for encoder in self.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, self.dropout, training=self.training)

            # Graph Output
            output = self.encoder.graph_encoders[-1](node_vec, cur_adj)
            kld = torch.zeros((1,)).to(self.device)
            reco_loss = 0
            mu_ = torch.zeros((output.size(0), self.config['latent_dim'])).to(self.device)

            # sentence representation
            # sent_rep = torch.cat([graph_hid, arg1, arg2], dim=1)
            output = self.compute_output(output, bag_size)

        rec_features = (kld, reco_loss, mu_)

        return output, (init_adj, cur_raw_adj, cur_adj, raw_node_vec, init_node_vec, node_vec, node_mask), \
               rec_features
