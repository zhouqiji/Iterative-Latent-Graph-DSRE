import torch.nn as nn
import torch.nn.functional as F

from modules.embed import *
from modules.encoders_decoders import *
from modules.attention import *
from modules.ModifiedAdaptiveSoftmax import AdaptiveLogSoftmaxWithLoss
from text_graph import TextGraph


class BaseNet(nn.Module):

    def __init__(self, config, vocabs, device):
        """
        :param config: model configs
        :param vocabs: word vocabs
        :param device: gpu or cpu
        """
        super(BaseNet, self).__init__()

        self.in_drop = nn.Dropout(p=config['input_dropout'])
        self.out_drop = nn.Dropout(p=config['output_dropout'])
        self.device = device
        self.config = config

        self.PAD_id = vocabs['w_vocab'].word2id[vocabs['w_vocab'].PAD]
        self.EOS_id = vocabs['w_vocab'].word2id[vocabs['w_vocab'].EOS]
        self.SOS_id = vocabs['w_vocab'].word2id[vocabs['w_vocab'].SOS]
        self.UNK_id = vocabs['w_vocab'].word2id[vocabs['w_vocab'].UNK]

        self.w_embed = EmbedLayer(num_embeddings=vocabs['w_vocab'].n_word,
                                  embedding_dim=config['word_embed_dim'],
                                  pretrained=vocabs['w_vocab'].pretrained,
                                  ignore=vocabs['w_vocab'].word2id[vocabs['w_vocab'].PAD],
                                  mapping=vocabs['w_vocab'].word2id,
                                  freeze=config['freeze_words'])

        self.r_embed = EmbedLayer(num_embeddings=len(vocabs['r_vocab']),
                                  embedding_dim=config['rel_embed_dim'])

        self.p_embed = EmbedLayer(num_embeddings=vocabs['p_vocab'].n_pos,
                                  embedding_dim=config['pos_embed_dim'],
                                  ignore=vocabs['p_vocab'].pos2id[vocabs['p_vocab'].PAD])

        self.graph_encoder = TextGraph(config=config, num_rel=len(vocabs['r_vocab']))


        self.rel_flatten = nn.Flatten(1, -1)

        self.dim2rel = nn.Linear(in_features=config['graph_out_dim'], out_features=len(vocabs['r_vocab']))
        # self.dim2rel.weight = self.r_embed.embedding.weight

        # task loss
        self.task_loss = nn.BCEWithLogitsLoss(reduction='none')

