import argparse
import random

from helpers.io import *
import json
from helpers.vocabs import *


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_priors(file):
    # TODO: impl the trained graph priors
    pass


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


    ###################################################################################

    # Models



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'test'])
    parser.add_argument('--show_example', action='store_true', help='Show an example')
    args = parser.parse_args()
    main(args)
