import torch.nn as nn
import torch.nn.functional as F

from modules.embed import *
from modules.encoders_decoders import *
from modules.attention import *
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

        self.w_embed = EmbedLayer(num_embeddings=vocabs['w_vocab'].n_word,
                                  embedding_dim=config['word_embed_dim'],
                                  pretrained=vocabs['w_vocab'].pretrained,
                                  ignore=vocabs['w_vocab'].word2id[vocabs['w_vocab'].PAD],
                                  mapping=vocabs['w_vocab'].word2id,
                                  freeze=config['freeze_words'])

        self.p_embed = EmbedLayer(num_embeddings=vocabs['p_vocab'].n_pos,
                                  embedding_dim=config['pos_embed_dim'],
                                  ignore=vocabs['p_vocab'].pos2id[vocabs['p_vocab'].PAD])

        self.graph_encoder = TextGraph(config=config, vocabs=vocabs, num_rel=len(vocabs['r_vocab']))

        # task loss
        self.task_loss = nn.BCEWithLogitsLoss(reduction='none')
