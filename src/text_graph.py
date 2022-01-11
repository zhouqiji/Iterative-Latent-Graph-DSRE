import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.gnn import GCN
from modules.graphlearn import GraphLearner, get_binarized_kneighbors_graph

from helpers.constants import INF, VERY_SMALL_NUMBER
from helpers.common import *


class TextGraph(nn.Module):
    def __init__(self, config, lang_encoder=None):
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

        # Text Sentence Embedding
        self.ctx_encoder = lang_encoder

        self.linear_out = nn.Linear(self.enc_dim, self.graph_out_dim)

        if self.graph_module == 'gcn':
            gcn_module = GCN
            self.encoder = gcn_module(nfeat=self.graph_hid_dim,
                                      nhid=self.graph_hid_dim,
                                      nclass=self.graph_hid_dim,
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

        else:
            self.graph_learner = None

    def compute_no_gnn_output(self, context, context_lens):
        context_vec = self.ctx_encoder(context, len_=context_lens)
        return context_vec

    def prepare_init_graph(self, context, context_lens):
        context_mask = create_mask(context_lens, context.size(-1), device=self.device)
        raw_context_vec = context

        context_vec = self.ctx_encoder(raw_context_vec, context_lens)

        init_adj = self.compute_init_adj(raw_context_vec.detach(), self.config['input_graph_knn_size'],
                                         mask=context_mask)
        return raw_context_vec, context_vec, context_mask, init_adj

    def forward(self, context_vec, context_len):
        # Prepare init node embedding, init adj
        raw_context_vec, context_vec, context_mask, init_adj = self.prepare_init_graph(context_vec, context_len)

        # Init
        raw_node_vec = raw_context_vec  # word embedding
        init_node_vec = context_vec  # hidden embedding
        node_mask = context_mask

