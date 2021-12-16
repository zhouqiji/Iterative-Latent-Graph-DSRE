from modules.embed import *
from modules.encoders_decoders import *
from modules.attention import *


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

        if config['include_positions']:
            input_dim = config['word_embed_dim'] + 2 * config['pos_embed_dim']
        else:
            input_dim = config['word_embed_dim']

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

        self.lang_encoder = LSTMEncoder(in_features=input_dim,
                                        h_enc_dim=config['enc_dim'],
                                        layers_num=config['enc_layers'],
                                        dir2=config['enc_bidirectional'],
                                        device=self.device,
                                        action='sum')

        # TODO: Selective Attention, Needed?
        self.sentence_attention = SelectiveAttention(device=self.device)
        self.dim2rel = nn.Linear(in_features=config['rel_embed_dim'], out_features=len(vocabs['r_vocab']))
        self.dim2rel.weight = self.r_embed.embedding.weight
