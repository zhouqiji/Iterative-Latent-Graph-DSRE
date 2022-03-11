import json

from helpers.io import *
from helpers.vocabs import *
from helpers.datasets import BagREDataset
from torch.utils.data import DataLoader
from helpers.collates import *

import torch
from bag_trainer import Trainer
from graph_net import GraphNet as Target_Model


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_priors(file):
    print("Loading mu vectors for pairs: {}".format(file))
    with open(file, 'r', encoding='utf-8') as infile:
        priors = json.load(infile)
    return priors


def load_data(word_vocab, prior_mus, config, pos_vocab=None, mode='train'):
    train_data_ = BagREDataset(config,
                               config['train_data'],
                               config['relations_file'],
                               word_vocab,
                               prior_mus,
                               pos_vocab=pos_vocab,
                               max_sent_length=config['max_sent_len'],
                               max_vocab=config['max_vocab_size'],
                               max_bag_size=config['bag_size'], mode=mode)
    print(len(train_data_))
    train_loader_ = DataLoader(dataset=train_data_,
                               batch_size=config['batch_size'],
                               shuffle=True,
                               collate_fn=BagCollates(),
                               num_workers=0)
    val_data_ = BagREDataset(config,
                             config['val_data'],
                             config['relations_file'],
                             train_data_.word_vocab if mode == 'train' else word_vocab,
                             prior_mus,
                             pos_vocab=train_data_.pos_vocab if mode == 'train' else pos_vocab,
                             max_sent_length=train_data_.max_sent_length, max_bag_size=0,
                             mode='val')
    print(len(val_data_))
    val_loader_ = DataLoader(dataset=val_data_,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             collate_fn=BagCollates(),
                             num_workers=0)

    test_data_ = BagREDataset(config,
                              config['test_data'],
                              config['relations_file'],
                              train_data_.word_vocab if mode == 'train' else word_vocab,
                              prior_mus,
                              pos_vocab=train_data_.pos_vocab if mode == 'train' else pos_vocab,
                              max_sent_length=train_data_.max_sent_length, max_bag_size=0,
                              mode='test')
    print(len(test_data_))
    test_loader_ = DataLoader(dataset=test_data_,
                              batch_size=config['batch_size'],
                              shuffle=False,
                              collate_fn=BagCollates(),
                              num_workers=0)

    print('Vocabulary size:         {}'.format(train_data_.word_vocab.n_word))
    print('Maximum sentence length:  {}'.format(train_data_.max_sent_length))

    return train_loader_, val_loader_, test_loader_, train_data_


def load_trainer(train_loader_, val_loader_, test_loader_, train_data_, config, device):
    trainer = Trainer(config, device,
                      iterators={'train': train_loader_, 'val': val_loader_, 'test': test_loader_},
                      vocabs={'w_vocab': train_data_.word_vocab, 'r_vocab': train_data_.rel_vocab,
                              'p_vocab': train_data_.pos_vocab})
    trainer.model = trainer.init_model(Target_Model)
    trainer.optimizer = trainer.set_optimizer(trainer.model)

    return trainer


def load_saved_model(config, prior_mus, device, which=None):
    trainer = Trainer(config, device,
                      iterators={'train': [], 'val': [], 'test': []},
                      vocabs={'w_vocab': {}, 'r_vocab': {}, 'p_vocab': {}})

    checkpoint = trainer.load_checkpoint(which)
    vocabs = checkpoint['vocabs']
    train_loader_, val_loader_, test_loader_, train_data_ = load_data(vocabs['w_vocab'], prior_mus, config,
                                                                      pos_vocab=vocabs['p_vocab'],
                                                                      mode='train-test')
    trainer.iterators['train'] = train_loader_
    trainer.iterators['val'] = val_loader_
    trainer.iterators['test'] = test_loader_
    trainer.iterations = len(train_loader_)

    trainer.model = trainer.init_model(Target_Model)
    trainer.optimizer = trainer.set_optimizer(trainer.model)
    trainer.assign_model(checkpoint)
    return trainer


def main(args):
    config = load_config(args.config)

    print("Setting the seed to {}".format(config['seed']))
    set_seed(config['seed'])

    config['mode'] = args.mode
    config['show_example'] = args.show_example
    config['model_folder'], config['exp'] = setup_log(config, mode=config['mode'], folder_name=config['exp_name'])
    device = torch.device("cuda:{}".format(config['device']) if config['device'] != -1 else "cpu")
    print()

    #  Pre-trained embeddings
    if config['pretrained_embeds_file']:
        print("Loading pretrained word embeddings ... ", end='')
        word_vocab = Words()
        word_vocab.pretrained = load_pretrained_embeds(config['pretrained_embeds_file'], config['word_embed_dim'])
    else:
        word_vocab = None

    # Priors
    if config['priors']:
        prior_mus = load_priors(config['prior_mus_file'])
    else:
        prior_mus = None

    # -----------------------------------------------------------------------------------

    # Models

    if config['mode'] == 'train':
        train_loader, val_loader, test_loader, train_data = load_data(word_vocab, prior_mus, config)
        trainer = load_trainer(train_loader, val_loader, test_loader, train_data, config, device)
        _ = trainer.run()

    elif config['mode'] == 'test':
        trainer = load_saved_model(config, prior_mus, device)
        print_options(config)

        for name_ in ['val', 'test']:
            tracker, time_ = trainer.eval_epoch(iter_name=name_)
            perf, p_points, r_points = trainer.calculate_performance(0, tracker, time_, mode=name_)
            np.savez(os.path.join(config['model_folder'], f'{name_}_pr.npz'), precision=p_points, recall=r_points)

            print(f'\n--- {name_} set performance ---')
            print('AUC: {:.4f}\np@100: {:.4f}, p@200: {:.4f}, p@300: {:.4f}, p@500: {:.4f}\n'.format(
                perf['pr_auc'], perf['p@100'], perf['p@200'], perf['p@300'], perf['p@500']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'test'])
    parser.add_argument('--show_example', action='store_true', help='Show an example')
    args = parser.parse_args()
    main(args)
