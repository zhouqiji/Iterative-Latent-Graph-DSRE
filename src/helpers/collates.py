import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class BaseCollate:
    """
    Base Class
    """

    def __init__(self, batch_first=True):
        self.batch_first = batch_first

    def pad_samples(self, samples, padding_value=0):
        return pad_sequence([torch.LongTensor(x) for x in samples], self.batch_first,
                            padding_value=padding_value)

    def _collate(self, *args):
        raise NotImplementedError

    def __call__(self, batch):
        return self._collate(batch)


class SentenceCollates(BaseCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, data):
        """
        allow dynamic padding based on the current batch
        """
        data = list(zip(*data))

        input_seqs = self.pad_samples(data[0])
        target_seqs = self.pad_samples(data[1])
        length = torch.tensor(list(data[2])).long()
        return {'source': input_seqs, 'target': target_seqs, 'sent_len': length}


class BagCollates(BaseCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, data):
        """
        allow dynamic padding based on the current batch
        """
        data = list(zip(*data))

        label, bag_name, bag_sizes, bag_ent_offsets = data[:4]
        labels = torch.from_numpy(np.stack(label)).long()
        bag_sizes = torch.from_numpy(np.array(bag_sizes)).long()

        bs = sum(data[4], [])
        p1 = sum(data[6], [])
        p2 = sum(data[7], [])
        slen = sum(data[8], [])
        mentions = sum(data[9], [])
        source_txt = sum(data[11], [])

        batch_seqs = self.pad_samples(bs)
        pos1 = self.pad_samples(p1)
        pos2 = self.pad_samples(p2)
        slen = torch.from_numpy(np.stack(slen)).long()
        mentions = torch.from_numpy(np.stack(mentions)).long()
        assert torch.sum(bag_sizes) == batch_seqs.size(0) == slen.size(0)

        # Get mentino adj
        mention_adj = torch.zeros([batch_seqs.size(0), batch_seqs.size(-1), batch_seqs.size(-1)])
        # Connected mention 1 with mention 2
        # for i in range(mention_adj.size(0)):
        #     m1_start, m1_end, m2_start, m2_end = mentions[i]
        #     for j in range(m1_start, m1_end + 1):
        #         for k in range(m2_start, m2_end + 1):
        #             mention_adj[i][j][k] = 1
        #
        #     for j in range(m2_start, m2_end + 1):
        #         for k in range(m1_start, m1_end + 1):
        #             mention_adj[i][j][k] = 1

        if all(d is None for d in data[10]):
            return {'rel': labels, 'bag_names': bag_name, 'bag_size': bag_sizes, 'source': batch_seqs,
                    'sent_len': slen, 'mentions': mentions,
                    'pos1': pos1, 'pos2': pos2, 'txt': source_txt}
        else:
            priors = torch.from_numpy(np.stack(data[10]))
            return {'rel': labels, 'bag_names': bag_name, 'bag_size': bag_sizes, 'source': batch_seqs,
                    'sent_len': slen, 'mentions': mentions,
                    'pos1': pos1, 'pos2': pos2, 'prior_mus': priors, 'txt': source_txt}
