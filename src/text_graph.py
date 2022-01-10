import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.gnn import GCN
from modules.graphlearn import GraphLearner, get_binarized_kneighbors_graph


class TextGraph(nn.Module):
    def __init__(self, config, in_dim):
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

        self.in_dim = in_dim
        self.graph_hid_dim = config['graph_hid_dim']
        self.graph_out_dim = config['graph_out_dim']

        # Text Sentence Embedding

        self.linear_out = nn.Linear(in_dim, self.graph_out_dim)

        if self.graph_module == 'gcn':
            gcn_module = GCN
            self.encoder = gcn_module(nfeat=in_dim,
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
        #TODO: If necessary?
        output = self.linear_out(context)
        return output

    def learn_graph(self):
        pass
