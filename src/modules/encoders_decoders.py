from .rnn import *

class LSTMEncoder(nn.Module):
    def __init__(self, in_features, h_enc_dim, layers_num, dir2, device, action='concat'):
        """
        :param in_features: input dim
        :param h_enc_dim: encoder hidden dim
        :param layers_num:  number of hidden layers
        :param dir2: bi-directional or not
        :param device: cpy/gpu
        :param action: sum or concat
        """
        super(LSTMEncoder, self).__init__()

        self.net = RNNLayer(input_size=in_features,
                            rnn_size=h_enc_dim,
                            num_layers=layers_num,
                            bidirectional=dir2)
        self.hidden = None
        self.dir2 = dir2
        self.h_enc_dim = h_enc_dim
        self.layers_num = layers_num
        self.device = device
        self.action = action

    def init_hidden(self, bs):
        if self.dir2:
            h_0 = torch.zeros(2 * self.layers_num, bs, self.h_enc_dim).to(self.device)
            c_0 = torch.zeros(2 * self.layers_num, bs, self.h_enc_dim).to(self.device)
        else:
            h_0 = torch.zeros(self.layers_num, bs, self.h_enc_dim).to(self.device)
            c_0 = torch.zeros(self.layers_num, bs, self.h_enc_dim).to(self.device)
        return h_0, c_0

    def keep_last_hidden(self, h_state, c_state):
        """
        Layers, directional, batch_size, dimension
        keep last layer
        """
        if self.dir2:
            h_state = h_state.view(self.layers_num, 2, -1, self.h_enc_dim)
            h_state = h_state[-1]
            if self.action == 'sum':
                h_state = torch.sum(h_state, 0)  # sum
            else:
                h_state = torch.cat(h_state.unbind(dim=0), dim=1)  # concatenation

            c_state = c_state.view(self.layers_num, 2, -1, self.h_enc_dim)
            c_state = c_state[-1]
            if self.action == 'sum':
                c_state = torch.sum(c_state, dim=0)
            else:
                c_state = torch.cat(c_state.unbind(dim=0), dim=1)
        else:
            h_state = h_state.view(self.layers_num, 1, -1, self.h_enc_dim)
            h_state = h_state[-1].squeeze(dim=0)

            c_state = c_state.view(self.layers_num, 1, -1, self.h_enc_dim)
            c_state = c_state[-1].squeeze(dim=0)
        return h_state, c_state

    def keep_output(self, output):
        if self.dir2:
            output = output.reshape(output.size(1), output.size(0), 2, self.h_enc_dim)

            if self.action == 'sum':
                output = output.sum(dim=2)  # sum
            else:
                output = torch.cat(output.unbind(dim=2), dim=2)  # concatenation
            output = output.view(output.size(1), output.size(0), output.size(2))
        return output

    def forward(self, x, len_=None):
        h_state = self.init_hidden(x.size(0))
        output, (h_state, c_state) = self.net(x, hidden=h_state, lengths=len_)

        h_state, c_state = self.keep_last_hidden(h_state, c_state)
        output = self.keep_output(output)
        return output, (h_state, c_state)


class LSTMDecoder(nn.Module):
    def __init__(self, in_features, h_dec_dim, layers_num, dir2, device, action='sum'):
        """
        :param in_features: input dimension
        :param h_dec_dim: decoder hidden dimension
        :param layers_num: number of hidden layers
        :param dir2:  bi-directional or not
        :param device: cpu/gpu
        :param action: sum or concat
        """
        super(LSTMDecoder, self).__init__()

        self.net = RNNLayer(input_size=in_features,
                            rnn_size=h_dec_dim,
                            num_layers=layers_num,
                            bidirectional=dir2)

        self.hidden = None
        self.dir2 = False
        self.layers_num = layers_num
        self.h_dec_dim = h_dec_dim
        self.device = device
        self.action = action

    def init_hidden(self, bs):
        h_0 = torch.zeros(self.layers_num, bs, self.h_dec_dim).to(self.device)
        c_0 = torch.zeros(self.layers_num, bs, self.h_dec_dim).to(self.device)
        return h_0, c_0

    def keep_last_hidden(self, h_state, c_state):
        """
         layers, directional, batch_size, dimension
         keep last layer
        """
        h_state = h_state.view(self.layers_num, 1, -1, self.h_dec_dim)
        c_state = h_state[-1].squeeze(dim=0)

        c_state = c_state.view(self.layers_num, 1, -1, self.h_dec_dim)
        c_state = c_state[-1].squeeze(dim=0)
        return h_state, c_state

    def forward(self, y, len_=None, hidden_=None):
        if hidden_ is None:
            h_state = self.init_hidden(y.size(0))
        else:
            h_state = hidden_

        output, (h_state, c_state) = self.net(y, hidden=h_state, lengths=len_)
        return output, (h_state, c_state)
