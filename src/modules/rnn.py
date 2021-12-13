import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNLayer(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, bidirectional, mode='lstm'):
        """
        :param input_size:  the size of the input features
        :param rnn_size: the size of the hidden state
        :param num_layers: number of stacked layers
        :param bidirectional: True/False
        :param mode: rnn, lstm, gru

        Returns: outputs, last_outputs
            (batch, seq_len, hidden_size):
                tensor containing the output features (h_t)
                from the last layer of the LSTM for each t.
            (batch, hidden_size):
                tensor containing the last output features
                from the last layer of the LSTM, for each t=seq_len
        """

        super(RNNLayer, self).__init__()

        self.bidirectional = bidirectional
        self.rnn_size = rnn_size

        if mode == 'gru':
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=rnn_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=rnn_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=True)

        # define output feature size
        self.feature_size = rnn_size
        if bidirectional:
            self.feature_size += 2

    @staticmethod
    def reorder_hidden(hidden, order):
        # hidden is a tuple (h_t, c_t)
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, order, :], hidden[1][:, order, :]
        else:
            hidden = hidden[:, order, :]
        return hidden

    def forward(self, x, hidden=None, lengths=None):
        batch, max_length, feat_size = x.size()

        if type(lengths) is not None:
            # sorting
            lengths_sorted, sorted_i = lengths.sort(descending=True)
            _, reverse_i = sorted_i.sort()

            x = x[sorted_i]
            if hidden is not None:
                hidden = self.reorder_hidden(hidden, sorted_i)

            # forward
            packed = pack_padded_sequence(x, lengths_sorted, batch_first=True)
            self.rnn.flatten_parameters()
            out_packed, hidden = self.rnn(packed, hidden)

            out_unpacked, _lengths = pad_packed_sequence(out_packed,
                                                         batch_first=True,
                                                         total_length=max_length)

            # un-sorting
            outputs = out_unpacked[reverse_i]
            hidden = self.reorder_hidden(hidden, reverse_i)
        else:
            self.rnn.flatten_parameters()
            outputs, hidden = self.rnn(x, hidden)
        return outputs, hidden
