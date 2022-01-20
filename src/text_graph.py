import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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

        # Text Sentence Embedding
        self.ctx_encoder = lang_encoder

        self.linear_out = nn.Linear(self.graph_out_dim, self.output_rel_dim)

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

    def forward(self, context_vec, context_len, mentions):
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

        node_vec = torch.relu(self.encoder.graph_encoders[0](init_node_vec, cur_adj))
        node_vec = F.dropout(node_vec, self.dropout, training=self.training)

        # Add mid GNN layers
        for encoder in self.encoder.graph_encoders[1:-1]:
            node_vec = torch.relu(encoder(node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.dropout, training=self.training)

        # Graph Output
        output = self.encoder.graph_encoders[-1](node_vec, cur_adj)
        # output = self.compute_output(output, node_mask=node_mask)
        # hidden = self.graph_maxpool(output.transpose(-1, -2), node_mask=node_mask)
        # hidden = F.dropout(hidden, self.dropout, training=self.training)

        # hidden = self.compute_output(output, node_mask=node_mask)
        # TODO: Complete hidden
        return output, (init_adj, cur_raw_adj, cur_adj, raw_node_vec, init_node_vec, node_vec, node_mask)
