import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from init_net import BaseNet
import numpy as np
from modules.constants import *

from text_graph import TextGraph


class GraphNet(BaseNet):
    """
    The main model
    """

    def greedy_decoding(self, z):
        h0 = self.latent2hid(z).unsqueeze(0)
        h0 = h0.expand(self.config['dec_layers'], h0.size(1), h0.size(2)).contiguous()
        c0 = torch.zeros(self.config['dec_layers'], z.size(0), self.config['dec_dim']).to(self.device).contiguous()

        # start with start-of-sentence (SOS)
        w_id = torch.empty((1,)).fill_(self.w_vocab.word2id[self.w_vocab.SOS]).to(self.device).long()
        gen_sentence = [self.w_vocab.SOS]

        while (gen_sentence[-1] != self.w_vocab.EOS) and (len(gen_sentence) <= self.config['max_sent_len']):
            dec_input = self.w_embed(w_id)
            dec_input = torch.cat([dec_input.unsqueeze(0), z.unsqueeze(0)], dim=2)

            next_word_rep, (h0, c0) = self.lang_decoder(dec_input, hidden_=(h0, c0))

            logits = self.reco_loss.log_prob(next_word_rep.squeeze(0))
            norm_logits = F.softmax(logits, dim=1)

            # w_id = torch.multinomial(norm_logits.squeeze(0), 1)
            w_id = norm_logits.argmax(dim=1)
            gen_sentence += [self.w_vocab.id2word[w_id.item()]]

        gen_sentence = ' '.join(gen_sentence[1:-1])
        print(gen_sentence + '\n')

    def sample_posterior(self, batch):
        x_vec = self.w_embed(batch['source'])  # (all-batch-sents, words, dim)

        if self.config['include_positions']:
            pos1 = self.p_embed(batch['pos1'])
            pos2 = self.p_embed(batch['pos2'])
            x_vec = torch.cat([x_vec, pos1, pos2], dim=2)
        x_vec = self.in_drop(x_vec)

        enc_out, (hidden, cell_state) = self.lang_encoder(x_vec, len_=batch['sent_len'])  # encoder

        new_input = torch.cat([hidden, cell_state], dim=1)
        mu_ = self.hid2mu(new_input)  # use sentence representation for reconstruction
        logvar_ = self.hid2var(new_input)
        latent_z1 = self.reparameterisation(mu_, logvar_)
        latent_z2 = self.reparameterisation(mu_, logvar_)
        latent_z3 = self.reparameterisation(mu_, logvar_)

        names = list(batch['bag_names'])
        r = np.repeat(np.arange(len(names)), repeats=batch['bag_size'].cpu().tolist())

        for i, (m, z1, z2, z3) in enumerate(zip(mu_, latent_z1, latent_z2, latent_z3)):  # for each sentence
            all_w_ids = [self.w_vocab.id2word[w_.item()] for w_ in batch['source'][i] if w_.item() != 0]

            arg1 = ' '.join(all_w_ids[batch['mentions'][i][0]:batch['mentions'][i][1] + 1])
            arg2 = ' '.join(all_w_ids[batch['mentions'][i][2]:batch['mentions'][i][3] + 1])

            if len(all_w_ids) <= 20 and ('NA' != names[r[i]].split(' ### ')[2]):
                print(' '.join(all_w_ids))
                print(arg1, '#', arg2, '#', names[r[i]])
                print('=' * 50)
                print('MEAN = ', end='')
                self.greedy_decoding(m.unsqueeze(0))
                print('SAMPLE 1 = ', end='')
                self.greedy_decoding(z1.unsqueeze(0))
                print('SAMPLE 2 = ', end='')
                self.greedy_decoding(z2.unsqueeze(0))
                print('SAMPLE 3 = ', end='')
                self.greedy_decoding(z3.unsqueeze(0))
                print('-' * 50)

    def homotomies(self):
        print(10 * '=', 'Generating homotomies', 10 * '=')

        z1 = torch.randn([1, self.config['latent_dim']]).to(self.device)  # random sample
        z2 = torch.randn([1, self.config['latent_dim']]).to(self.device)
        z = 0.2 * z1 + 0.8 * z2

        self.greedy_decoding(z1)
        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            z = i * z1 + (1 - i) * z2
            self.greedy_decoding(z)
        self.greedy_decoding(z2)

    def generation(self, examples_num=10):
        """
        Generate sentences given a random sample
        """
        print(10 * '=', 'Generating sentences', 10 * '=')
        for i in range(examples_num):
            z = torch.randn([1, self.config['latent_dim']]).to(self.device)
            self.greedy_decoding(z)

    # ----------------------------------------------- #
    # TODO: ALL for Reconstruction, need to be modified to graph version
    def reparameterisation(self, mean_, logvar_):
        std = torch.exp(0.5 * logvar_)
        eps = torch.randn_like(std)
        return mean_ + (eps * std)

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

    def form_decoder_input(self, words):
        """ Word dropout: https://www.aclweb.org/anthology/K16-1002/ """

        random_z2o = torch.rand(words.size()).to(self.device)
        cond1 = torch.lt(random_z2o, self.config['teacher_force'])  # if < word_drop
        cond2 = torch.ne(words, self.PAD_id) & \
                torch.ne(words, self.SOS_id)  # if != PAD & SOS

        dec_input = torch.where(cond1 & cond2,
                                torch.full_like(words, self.UNK_id),
                                words)
        dec_input = self.w_embed(dec_input)
        return dec_input

    def reconstruction(self, latent_z, batch):
        y_vec = self.form_decoder_input(batch['source'])
        y_vec = torch.cat([y_vec,
                           latent_z.unsqueeze(dim=1).repeat((1, y_vec.size(1), 1))], dim=2)

        h_0 = self.latent2hid(latent_z).unsqueeze(0)
        h_0 = h_0.expand(self.config['dec_layers'], h_0.size(1), h_0.size(2))
        c_0 = torch.zeros(self.config['dec_layers'], latent_z.size(0), self.config['dec_dim']).to(self.device)

        recon_x, _ = self.lang_decoder(y_vec, len_=batch['sent_len'], hidden_=(h_0, c_0))
        return recon_x

    def calc_reconstruction_loss(self, recon_x, batch):
        # remove padded
        tmp = torch.arange(recon_x.size(1)).repeat(batch['sent_len'].size(0), 1).to(self.device)
        mask = torch.lt(tmp, batch['sent_len'][:, None].repeat(1, tmp.size(1)))  # result in (words, dim)

        # Convert to (sentences, words)
        o_vec = self.reco_loss(recon_x[mask], batch['target'][mask])  # (words,)
        o_vec = pad_sequence(torch.split(o_vec.loss, batch['sent_len'].tolist(), dim=0),
                             batch_first=True,
                             padding_value=0)

        mean_mean = torch.div(torch.sum(o_vec, dim=1), batch['sent_len'].float().to(self.device))
        reco_loss = {'mean': torch.mean(mean_mean),  # mean over words, mean over batch (for perplexity)
                     'sum': torch.mean(torch.sum(o_vec, dim=1))}  # sum over words, mean over batch (sentences)
        return reco_loss

    @staticmethod
    def calc_kld(mu, logvar, mu_prior=None, logvar_prior=None):
        if mu_prior is not None:
            mu_diff = mu_prior.float() - mu
            kld = -0.5 * (1 + logvar - mu_diff.pow(2) - logvar.exp())
        else:
            kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        kld = torch.sum(kld, dim=1)
        kld = torch.mean(kld)  # sum over dim, mean over batch
        return kld

    def calc_task_loss(self, rel_logits, target):
        task_loss = self.task_loss(rel_logits, target.type_as(rel_logits))  # (examples, categories)
        task_loss = torch.sum(task_loss, dim=1)
        task_loss = torch.mean(task_loss)  # sum across relations, mean over batch
        rel_probs = torch.sigmoid(rel_logits)

        # Get the score

        return rel_probs, task_loss

    def add_batch_graph_loss(self, out_adj, features, keep_batch_dim=False):
        # Graph regularization
        if keep_batch_dim:
            graph_loss = []
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss.append(self.config['smoothness_ratio'] * torch.trace(
                    torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(
                    np.prod(out_adj.shape[1:])))

            graph_loss = torch.Tensor(graph_loss).to(self.device)

            ones_vec = torch.ones(out_adj.shape[:-1], device=self.device)
            graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(
                torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).squeeze(-1).squeeze(-1) / \
                          out_adj.shape[-1]
            graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2), (1, 2)) / int(
                np.prod(out_adj.shape[1:]))

        else:
            graph_loss = 0
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss += self.config['smoothness_ratio'] * torch.trace(
                    torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape))

            ones_vec = torch.ones(out_adj.shape[:-1], device=self.device)
            graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(
                torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).sum() / out_adj.shape[0] / \
                          out_adj.shape[-1]
            graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))

        return graph_loss

    def SquaredFrobeniusNorm(self, X):
        return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))

    def batch_SquaredFrobeniusNorm(self, X):
        return torch.sum(torch.pow(X, 2), (1, 2)) / int(np.prod(X.shape[1:]))

    def batch_diff(self, X, Y, Z):
        assert X.shape == Y.shape
        diff_ = torch.sum(torch.pow(X - Y, 2), (1, 2))
        norm_ = torch.sum(torch.pow(Z, 2), (1, 2))
        diff_ = diff_ / torch.clamp(norm_, min=VERY_SMALL_NUMBER)
        return diff_

    # model forward
    def forward(self, batch):
        ######################
        # Encoder
        ######################
        x_vec = self.w_embed(batch['source'])

        if self.config['include_positions']:
            pos1 = self.p_embed(batch['pos1'])
            pos2 = self.p_embed(batch['pos2'])
            x_vec = torch.cat([x_vec, pos1, pos2], dim=2)

        x_vec = self.in_drop(x_vec)

        ##########################
        # Graph Encoder
        ##########################
        graph_out, graph_hid, graph_features = self.graph_encoder(x_vec, batch['sent_len'], batch['mentions'])

        ##########################
        # Argument Representation
        ##########################

        # arg1, arg2 = self.merge_tokens(graph_out, batch['mentions'])  # contextualised representations of argument

        #####################
        # Reconstruction
        #####################
        if self.config['reconstruction']:

            new_input = torch.cat([graph_hid, graph_hid], dim=1)

            # create hidden code
            mu_ = self.hid2mu(new_input)
            logvar_ = self.hid2var(new_input)
            latent_z = self.reparameterisation(mu_, logvar_)

            if self.config['priors']:
                prior_mus_expanded = torch.repeat_interleave(batch['prior_mus'], repeats=batch['bag_size'], dim=0)
                kld = self.calc_kld(mu_, logvar_, mu_prior=prior_mus_expanded)

            else:
                kld = self.calc_kld(mu_, logvar_)

            # reconstruction
            recon_x = self.reconstruction(latent_z, batch)
            reco_loss = self.calc_reconstruction_loss(recon_x, batch)

            # sentence representation --> use info from VAE !!
            sent_rep = torch.cat([latent_z, latent_z], dim=1)
            sent_rep = pad_sequence(torch.split(sent_rep, batch['bag_size'].tolist(), dim=0),
                                    batch_first=True,
                                    padding_value=0)
            sent_rep = self.graph_encoder.graph_maxpool(sent_rep.transpose(-1, -2))
            sent_rep = self.graph_encoder.linear_out(sent_rep)

        else:
            kld = torch.zeros((1,)).to(self.device)
            reco_loss = {'sum': torch.zeros((1,)).to(self.device),
                         'mean': torch.zeros((1,)).to(self.device)}
            mu_ = torch.zeros((graph_out.size(0), self.config['latent_dim'])).to(self.device)

            # sentence representation
            # sent_rep = torch.cat([graph_hid, arg1, arg2], dim=1)
            sent_rep = self.graph_encoder.compute_output(graph_out, batch['bag_size'])

        # ######################
        # ## Classification
        # ######################

        rel_probs, task_loss = self.calc_task_loss(sent_rep, batch['rel'])

        init_adj, cur_raw_adj, cur_adj, raw_node_vec, init_node_vec, node_vec, node_mask = graph_features

        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            task_loss += self.add_batch_graph_loss(cur_raw_adj, raw_node_vec)

        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        # Simper version
        if self.training:
            max_iter = self.config['graph_learn_max_iter']
        else:
            max_iter = self.config['graph_learn_max_iter'] * 2

        eps_adj = float(self.config['eps_adj'])

        # For graph learning
        loss = 0
        iter_ = 0

        # Indicate the last iteration umber for each example
        batch_last_iters = torch.zeros(batch['source'].size(0), dtype=torch.uint8, device=self.device)
        # Indicate either an xample is in ongoing state (i.e., 1) or stopping state (i.e., 0)
        batch_stop_indicators = torch.ones(batch['source'].size(0), dtype=torch.uint8, device=self.device)
        batch_all_outputs = []

        while self.config['graph_learn'] and (
                iter_ == 0 or torch.sum(batch_stop_indicators).item() > 0) and iter_ < max_iter:
            iter_ += 1
            batch_last_iters += batch_stop_indicators
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = self.graph_encoder.learn_graph(self.graph_encoder.graph_learner2, node_vec,
                                                                  self.graph_encoder.graph_skip_conn,
                                                                  node_mask=node_mask,
                                                                  graph_include_self=self.graph_encoder.graph_include_self,
                                                                  init_adj=init_adj)

            update_adj_ratio = self.config['update_adj_ratio']
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj

            node_vec = torch.relu(self.graph_encoder.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.config['gl_dropout'], training=self.training)

            # Add mid GNN layers if needed
            for encoder in self.graph_encoder.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, self.config['gl_dropout'], training=self.training)

            tmp_output_sent = self.graph_encoder.encoder.graph_encoders[-1](node_vec, cur_adj)
            tmp_graph_hid = self.graph_encoder.graph_maxpool(tmp_output_sent.transpose(-1, -2))

            #####################
            # Reconstruction
            #####################
            if self.config['reconstruction']:
                tmp_new_input = torch.cat([tmp_graph_hid, tmp_graph_hid], dim=1)

                # create hidden code
                tmp_mu_ = self.hid2mu(tmp_new_input)
                tmp_logvar_ = self.hid2var(tmp_new_input)
                tmp_latent_z = self.reparameterisation(tmp_mu_, tmp_logvar_)

                if self.config['priors']:
                    tmp_prior_mus_expanded = torch.repeat_interleave(batch['prior_mus'], repeats=batch['bag_size'],
                                                                     dim=0)
                    tmp_kld = self.calc_kld(tmp_mu_, tmp_logvar_, mu_prior=tmp_prior_mus_expanded)

                else:
                    tmp_kld = self.calc_kld(tmp_mu_, tmp_logvar_)

                # reconstruction
                tmp_recon_x = self.reconstruction(tmp_latent_z, batch)
                tmp_reco_loss = self.calc_reconstruction_loss(tmp_recon_x, batch)

                # sentence representation --> use info from VAE !!

                tmp_output = torch.cat([tmp_latent_z, tmp_latent_z], dim=1)
                tmp_output = pad_sequence(torch.split(tmp_output, batch['bag_size'].tolist(), dim=0),
                                          batch_first=True,
                                          padding_value=0)
                tmp_output = self.graph_encoder.graph_maxpool(tmp_output.transpose(-1, -2))
                tmp_output = self.graph_encoder.linear_out(tmp_output)

            else:
                kld = torch.zeros((1,)).to(self.device)
                reco_loss = {'sum': torch.zeros((1,)).to(self.device),
                             'mean': torch.zeros((1,)).to(self.device)}
                mu_ = torch.zeros((graph_out.size(0), self.config['latent_dim'])).to(self.device)

                # sentence representation
                # tmp_output_sent = torch.cat([tmp_hidden, arg1, arg2], dim=1)
                tmp_output = self.graph_encoder.compute_output(tmp_output_sent, batch['bag_size'])

            # Sentence per bag
            # tmp_output = pad_sequence(torch.split(tmp_output_sent, batch['bag_size'].tolist(), dim=0),
            #                           batch_first=True,
            #                           padding_value=0)
            # TODO: simple version
            # tmp_output = self.reduction(tmp_output)
            # tmp_output = self.sentence_attention(tmp_output, batch['bag_size'], self.r_embed.embedding.weight.data)
            # tmp_output = self.graph_encoder.compute_output(tmp_output, self.dim2rel)

            #####################
            # Classification
            #####################
            # tmp_output = self.out_drop(tmp_output)
            # tmp_output = self.dim2rel(tmp_output)  # tie embeds
            # tmp_output = tmp_output.diagonal(dim1=1, dim2=2)  # take probs based on relations query vector
            batch_all_outputs.append(tmp_output_sent.unsqueeze(1))

            _, tmp_loss = self.calc_task_loss(tmp_output, batch['rel'])
            if len(tmp_loss.shape) == 2:
                tmp_loss = torch.mean(tmp_loss, 1)

            loss += batch_stop_indicators.float() * tmp_loss

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                tmp_graph_loss = self.add_batch_graph_loss(cur_raw_adj, raw_node_vec,
                                                           keep_batch_dim=True)
                loss += batch_stop_indicators.float() * tmp_graph_loss

            if self.config['graph_learn'] and not self.config['graph_learn_ratio'] in (None, 0):
                loss += batch_stop_indicators.float() * self.batch_SquaredFrobeniusNorm(cur_adj - pre_raw_adj) * \
                        self.config['graph_learn_ratio']
            tmp_stop_criteria = self.batch_diff(cur_raw_adj, pre_raw_adj, first_raw_adj) > eps_adj
            batch_stop_indicators = batch_stop_indicators * tmp_stop_criteria

        if iter_ > 0:
            loss = torch.mean(loss / batch_last_iters.float()) + task_loss

            batch_all_outputs = torch.cat(batch_all_outputs, 1)
            selected_iter_index = batch_last_iters.long().unsqueeze(-1) - 1
            if len(batch_all_outputs.shape) == 4:
                selected_iter_index = selected_iter_index.unsqueeze(-1).expand(-1, batch_all_outputs.size(-2),
                                                                               batch_all_outputs.size(-1)).unsqueeze(1)
                output = batch_all_outputs.gather(1, selected_iter_index).squeeze(1)
            else:
                output = batch_all_outputs.gather(1, selected_iter_index)

            output = self.graph_encoder.compute_output(output, batch['bag_size'])
            rel_probs, _ = self.calc_task_loss(output, batch['rel'])
        else:
            loss = task_loss

        assert torch.sum(torch.isnan(rel_probs)) == 0.0, sent_rep
        return task_loss, loss, rel_probs, kld, reco_loss, mu_
