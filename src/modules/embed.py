import torch
from torch import nn


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, ignore=None, freeze=False, pretrained=None, mapping=None):
        """
        :param num_embeddings: number of unique items
        :param embedding_dim: dimension of vectors
        :param ignore: None
        :param freeze: None
        :param pretrained: pretrained embeddings
        :param mapping: mapping of items of unique ids
        """

        super(EmbedLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=ignore)

        if pretrained is not None:
            print("Initialising with pre-trained word embeddings!")
            self.init_pretrained(pretrained, mapping)

        self.embedding.weight.requires_grad = not freeze

    def init_pretrained(self, pretrained, mapping):
        """
        :param pretrained: keys are words, values are vectors
        :param mapping:  keys are words, values are unique ids
        :return: update the embedding matrix with pretrained embeddings
        """
        found = 0
        for word in mapping.keys():  # words in vocabulary
            if word in pretrained:
                self.embedding.weight.data[mapping[word], :] = torch.from_numpy(pretrained[word])
                found += 1
            elif word.lower() in pretrained:
                self.embedding.weight.data[mapping[word], :] = torch.from_numpy(pretrained[word.lower()])
                found += 1

        print('Assigned {:.2f}% words a pretrained word embedding\n'.format(found * 100 / len(mapping)))

    def forward(self, xs):
        """
        :param xs: [batch, word_ids]
        :return: [batch, word_ids, dimension]
        """
        embeds = self.embedding(xs)
        return embeds
