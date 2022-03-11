import numpy as np
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from .vocabs import *
from collections import OrderedDict
from transformers import BertTokenizer

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


# Load bert tokenizer
def load_tokenizer(bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


class BagREDataset(Dataset):
    """
    Bag-level relation extraction dataset.
    Relation of NA should be named as 'NA'.
    """

    def __init__(self, config, path, rel2id, word_vocab, priors, pos_vocab=None, max_sent_length=None, max_vocab=None,
                 max_bag_size=0, bert_name=None, mode='train'):
        super().__init__()

        self.rel_vocab = json.load(open(rel2id, 'r', encoding='UTF-8')) if rel2id else Relations()
        self.word_vocab = word_vocab if word_vocab else Words()
        self.pos_vocab = pos_vocab if pos_vocab else Positions()

        self.max_bag_size = max_bag_size  # maximum bag size of the dataset
        self.max_sent_length = max_sent_length
        self.mode = mode
        self.max_vocab = max_vocab
        self.priors = priors
        if priors:
            self.priors_dim = len(self.priors[list(priors.keys())[-1]])

        # Berttokenizer
        self.config = config
        if self.config['using_bert']:
            self.tokenizer = load_tokenizer(bert_name)

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)
        self.data = []
        self.bag_scope = {}
        self.name2id, self.id2name = {}, {}
        self.bag_labels, self.bag_offsets = {}, {}
        self.avg_s_len, self.instances = [], 0

        self.unique_sentences = OrderedDict()
        with open(path, encoding='UTF-8') as infile:
            for item in tqdm(infile, desc='Loading ' + self.mode.upper()):
                sample = json.loads(item)

                bag_name = sample['bag_name']
                self.name2id[bag_name] = len(self.name2id)
                self.id2name[len(self.name2id) - 1] = bag_name
                self.bag_scope[bag_name] = {'text': [], 'offsets': [], 'labels': sample['bag_labels']}

                for s in sample['sentences']:
                    self.instances += 1
                    txt = s['text']
                    e1 = s['h']['tokens']
                    e2 = s['t']['tokens']
                    tokens_num = len(txt.split(' '))

                    if config['using_bert']:
                        temp_sent_list = txt.split()
                        # insert entity tag
                        if min(e1) > max(e2):  # e1 after e2
                            # insert e1
                            e1_start = e1[0]
                            temp_sent_list.insert(e1_start, "<e1>")
                            e1_end = e1[0] + len(e1) + 1
                            temp_sent_list.insert(e1_end, "</e1>")
                            # insert e2
                            e2_start = e2[0]
                            temp_sent_list.insert(e2_start, "<e2>")
                            e2_end = e2[0] + len(e2) + 1
                            temp_sent_list.insert(e2_end, "</e2>")
                            e1 = [temp_sent_list.index("<e1>"), temp_sent_list.index("</e1>")]
                            e2 = [temp_sent_list.index("<e2>"), temp_sent_list.index("</e2>")]
                        elif max(e1) < min(e2):  # e2 after e1
                            # insert e2
                            e2_start = e2[0]
                            temp_sent_list.insert(e2_start, "<e2>")
                            e2_end = e2[0] + len(e2) + 1
                            temp_sent_list.insert(e2_end, "</e2>")
                            # insert e1
                            e1_start = e1[0]
                            temp_sent_list.insert(e1_start, "<e1>")
                            e1_end = e1[0] + len(e1) + 1
                            temp_sent_list.insert(e1_end, "</e1>")
                            e1 = [temp_sent_list.index("<e1>"), temp_sent_list.index("</e1>")]
                            e2 = [temp_sent_list.index("<e2>"), temp_sent_list.index("</e2>")]
                        else:  # e1 or e2 contain others
                            if e2[0] <= e1[0] and e2[-1] >= e1[-1]:  # e2 contains e1
                                # insert e1
                                e1_start = e1[0]
                                temp_sent_list.insert(e1_start, "<e1>")
                                e1_end = e1[0] + len(e1) + 1
                                temp_sent_list.insert(e1_end, "</e1>")
                                # insert e2
                                e2_start = e2[0]
                                temp_sent_list.insert(e2_start, "<e2>")
                                e2_end = e2[0] + len(e2) + (e1_end - e1_start) + 1
                                temp_sent_list.insert(e2_end, "</e2>")
                                e2 = e1 = [temp_sent_list.index("<e2>"), temp_sent_list.index("</e2>")]
                            else:  # e1 contains e2
                                # insert e2
                                e2_start = e2[0]
                                temp_sent_list.insert(e2_start, "<e2>")
                                e2_end = e2[0] + len(e2) + 1
                                temp_sent_list.insert(e2_end, "</e2>")
                                # insert e1
                                e1_start = e1[0]
                                temp_sent_list.insert(e1_start, "<e1>")
                                e1_end = e1[0] + len(e1) + (e2_end - e2_start) + 1
                                temp_sent_list.insert(e1_end, "</e1>")
                                e1 = e2 = [temp_sent_list.index("<e1>"), temp_sent_list.index("</e1>")]

                        # e1 = [temp_sent_list.index("<e1>"), temp_sent_list.index("</e1>")]
                        # e2 = [temp_sent_list.index("<e2>"), temp_sent_list.index("</e2>")]
                        e1 = list(range(e1[0], e1[-1] + 1))
                        e2 = list(range(e2[0], e2[-1] + 1))
                        tokens_num = len(temp_sent_list)
                        if len(e1) == 1:
                            e1_ = [e1[0]] * tokens_num
                        else:
                            e1_ = [e1[0]] * (e1[0]) + e1 + [e1[-1]] * (tokens_num - e1[-1] - 1)

                        if len(e2) == 1:
                            e2_ = [e2[0]] * tokens_num
                        else:
                            e2_ = [e2[0]] * (e2[0]) + e2 + [e2[-1]] * (tokens_num - e2[-1] - 1)

                        assert len(e1_) == tokens_num and len(e2_) == tokens_num
                        txt = " ".join(temp_sent_list)
                    else:

                        if len(e1) == 1:
                            e1_ = [e1[0]] * tokens_num
                        else:
                            e1_ = [e1[0]] * (e1[0]) + e1 + [e1[-1]] * (tokens_num - e1[-1] - 1)

                        if len(e2) == 1:
                            e2_ = [e2[0]] * tokens_num
                        else:
                            e2_ = [e2[0]] * (e2[0]) + e2 + [e2[-1]] * (tokens_num - e2[-1] - 1)

                        assert len(e1_) == tokens_num and len(e2_) == tokens_num

                    pos1 = np.array(range(tokens_num), 'i') - np.array(e1_)
                    pos2 = np.array(range(tokens_num), 'i') - np.array(e2_)

                    self.unique_sentences[txt] = 1
                    self.avg_s_len += [tokens_num]
                    self.bag_scope[bag_name]['text'] += [txt]
                    self.bag_scope[bag_name]['offsets'] += [{'m1': e1, 'm2': e2, 'pos1': pos1, 'pos2': pos2}]

        print('Unique sentences: ', len(self.unique_sentences))
        if self.mode == 'train':
            self.make_vocabs()

        print('# instances:  {}\n# bags:        {}\n# relations:   {}'.format(self.instances, len(self.bag_scope),
                                                                              len(self.rel_vocab)))

        # Process data to tensors
        self.make_dataset()
        print()

    def make_vocabs(self):
        """
        Construct vocabularies
        """
        for us in self.unique_sentences:
            self.word_vocab.add_sentence(us)
        print('Avg. sent length: {:.04}'.format(sum(self.avg_s_len) / len(self.avg_s_len)))

        for pos in range(-self.max_sent_length, self.max_sent_length + 1):
            self.pos_vocab.add_position(pos)

        tot_voc = len(self.word_vocab.word2id)
        if self.max_vocab:
            print('Coverage:        {} %'.format(self.word_vocab.coverage(self.max_vocab)))
            self.word_vocab.resize_vocab_maxsize(self.max_vocab)

        print('Total vocab size: {}/{}'.format(self.word_vocab.n_word, tot_voc))

    def make_dataset(self):
        """
        Convert dataset to tensors
        """
        skipped = 0
        for i in tqdm(range(len(self.bag_scope)), desc='Processing'):
            pair_name = self.id2name[i]
            bag_sents = self.bag_scope[pair_name]['text']
            bag_ent_offsets = self.bag_scope[pair_name]['offsets']
            bag_label = list(set(self.bag_scope[pair_name]['labels']))
            assert len(bag_sents) == len(bag_ent_offsets)

            # binarize labels
            labels = np.zeros((len(self.rel_vocab),), 'i')
            for rel in bag_label:
                labels[self.rel_vocab[rel]] = 1

            bag_seqs, bag_seqs_target, pos1, pos2, sent_len, bag_mentions, attn_mask, token_ids = [], [], [], [], [], [], [], []
            for sentence, mentions in zip(bag_sents, bag_ent_offsets):

                # TODO: Get Bert token and masks [input_ids, attention_masks, token_type_is]
                if self.config['using_bert']:
                    tokens_a = self.tokenizer.tokenize(sentence)
                    cls_token_id = self.tokenizer.cls_token_id

                    e11_p = tokens_a.index("<e1>")  # the start position of entity1
                    e12_p = tokens_a.index("</e1>")  # the end position of entity1
                    e21_p = tokens_a.index("<e2>")  # the start position of entity2
                    e22_p = tokens_a.index("</e2>")  # the end position of entity2

                    e1 = list(range(e11_p, e12_p + 1))
                    e2 = list(range(e21_p, e22_p + 1))
                    # Replace the token
                    tokens_a[e11_p] = "$"
                    tokens_a[e12_p] = "$"
                    tokens_a[e21_p] = "#"
                    tokens_a[e22_p] = "#"

                    # Update mentions
                    mentions['m1'] = e1
                    mentions['m2'] = e2
                    tokens_num = len(tokens_a)
                    if len(e1) == 1:
                        e1_ = [e1[0]] * tokens_num
                    else:
                        e1_ = [e1[0]] * (e1[0]) + e1 + [e1[-1]] * (tokens_num - e1[-1] - 1)

                    if len(e2) == 1:
                        e2_ = [e2[0]] * tokens_num
                    else:
                        e2_ = [e2[0]] * (e2[0]) + e2 + [e2[-1]] * (tokens_num - e2[-1] - 1)

                    assert len(e1_) == tokens_num and len(e2_) == tokens_num

                    p1 = np.array(range(tokens_num), 'i') - np.array(e1_)
                    p2 = np.array(range(tokens_num), 'i') - np.array(e2_)

                    mentions['pos1'] = p1
                    mentions['pos2'] = p2

                    tokens_a = self.tokenizer.convert_tokens_to_ids(tokens_a)

                    if self.mode == 'train' or self.mode == 'train-test':
                        tokens_a = tokens_a[:self.max_sent_length]  # restrict to max_sent_length add 'cls'

                    token_type_ids = [0] * len(tokens_a)
                    tokens_a = [cls_token_id] + tokens_a
                    token_type_ids = [0] + token_type_ids

                    # input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                    attention_mask = [1] * len(tokens_a)
                    tmp_source = tokens_a
                else:
                    tmp = self.word_vocab.get_ids(sentence, replace=False)
                    if self.mode == 'train' or self.mode == 'train-test':
                        tmp = tmp[:self.max_sent_length]  # restrict to max_sent_length
                    tmp_source = [self.word_vocab.word2id[self.word_vocab.SOS]] + tmp  # add <SOS>
                    attention_mask = [0] * len(tmp_source)
                    token_type_ids = [0] * len(tmp_source)

                bag_seqs += [torch.tensor(tmp_source).long()]

                attn_mask += [attention_mask]
                token_ids += [token_type_ids]
                sent_len += [len(tmp_source)]

                if self.config['using_bert']:
                    bag_mentions += [[mentions['m1'][0] + 1, mentions['m1'][-1] + 1,
                                      mentions['m2'][0] + 1, mentions['m2'][-1] + 1]]
                else:
                    bag_mentions += [[mentions['m1'][0] + 1, mentions['m1'][-1] + 1,
                                      mentions['m2'][0] + 1, mentions['m2'][-1] + 1]]

                # Reduce mentions greater than max_length
                if self.mode == 'train' or self.mode == 'train-test':
                    for m in bag_mentions:
                        for idx in range(len(m)):
                            if m[idx] > self.max_sent_length:
                                m[idx] = self.max_sent_length

                # Do not forget the additional <PAD> for <SOS>
                pos1_ = [self.pos_vocab.pos2id[self.pos_vocab.PAD]] + \
                        self.pos_vocab.get_ids(mentions['pos1'], self.max_sent_length)
                pos2_ = [self.pos_vocab.pos2id[self.pos_vocab.PAD]] + \
                        self.pos_vocab.get_ids(mentions['pos2'], self.max_sent_length)

                if self.mode == 'train' or self.mode == 'train-test':
                    pos1 += [torch.tensor(pos1_[:self.max_sent_length + 1]).long()]
                    pos2 += [torch.tensor(pos2_[:self.max_sent_length + 1]).long()]
                else:
                    pos1 += [torch.tensor(pos1_).long()]
                    pos2 += [torch.tensor(pos2_).long()]

            if self.priors:
                if pair_name in self.priors:
                    priors = np.asarray(self.priors[pair_name])
                    self.data.append([labels, pair_name + ' ### ' + bag_label[0], len(bag_sents), bag_ent_offsets,
                                      bag_seqs, bag_seqs_target, pos1, pos2, sent_len, bag_mentions, priors,
                                      self.config['using_bert'], attn_mask,
                                      token_ids])
                else:
                    priors = np.zeros((self.priors_dim,))
                    # if self.mode == 'train' or self.mode == 'train-test':
                    #     skipped += 1
                    #     continue
                    # else:
                    self.data.append([labels, pair_name + ' ### ' + bag_label[0], len(bag_sents), bag_ent_offsets,
                                      bag_seqs, bag_seqs_target, pos1, pos2, sent_len, bag_mentions, priors,
                                      attn_mask,
                                      token_ids])
            else:
                priors = None
                self.data.append([labels, pair_name + ' ### ' + bag_label[0], len(bag_sents), bag_ent_offsets,
                                  bag_seqs, bag_seqs_target, pos1, pos2, sent_len, bag_mentions, priors,
                                  attn_mask,
                                  token_ids])

        print('Skipped: {}'.format(skipped))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return_list = self.data[item]
        return return_list
