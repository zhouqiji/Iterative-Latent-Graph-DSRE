import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from sklearn.metrics import precision_recall_curve, average_precision_score, auc


def plot_pr(dataset):
    sns.set(style="whitegrid")
    sns.set_context("paper")
    fig = plt.figure()

    c = [u'#4878d0', u'#ee854a', u'#6acc64', u'#d65f5f', u'#956cb4', u'#8c613c', u'#dc7ec0', u'#797979', u'#d5bb67',
         u'#82c6e2']
    if dataset == 'nyt10':
        names = ['pcnnatt', 'reside', 'intra-inter', 'distre', 'recon-sent', 'ours']
        color = ['#797979', '#dc7ec0', '#d5bb67', '#6acc64', '#ee854a', '#d65f5f']
        marker = ['d', 's', '^', '*', 'v', 'o']
    elif dataset == 'nyt10_570k':
        names = ['pcnnatt', 'reside', 'intra-inter',  'recon-sent', 'ours']
        color = ['#797979', '#dc7ec0', '#d5bb67', '#6acc64', '#d65f5f']
        marker = ['d', 's', '^', 'v', 'o']
    else:
        names = ['pcnnatt', 'reside', 'intra-inter', 'distre', 'recon-sent', 'ours']
        color = ['#797979', '#dc7ec0', '#d5bb67', '#6acc64', '#ee854a', '#d65f5f']
        marker = ['d', 's', '^', '*', 'v', 'o']

    for i, name in enumerate(names):

        if dataset == 'nyt10':
            path = os.path.join('nyt10', '520K', name)
        elif dataset == 'nyt10_570k':
            path = os.path.join('nyt10', '570K', name)
        else:
            path = os.path.join('wikidistant', name)

        if name in ['distre', 'reside', 'intra-inter', 'jointnre']:
            prec = np.load(os.path.join('../pr_curves/', path, 'precision.npy'))
            rec = np.load(os.path.join('../pr_curves/', path, 'recall.npy'))

        elif name in ['pcnnatt']:
            points = np.load(os.path.join('../pr_curves/', path, 'nyt10_test_pr.npz'))
            prec = points['precision']
            rec = points['recall']

        elif name in ['recon-sent']:
            points = np.load(os.path.join('../pr_curves/', path, 'test_pr.npz'))
            prec = points['precision']
            rec = points['recall']
        elif name in ['ours']:
            points = np.load(os.path.join('../pr_curves/', path, 'test_pr.npz'))
            prec = points['precision']
            rec = points['recall']

        else:
            rec = 0
            prec = 0

        area = auc(rec, prec)
        print(f'Name: {name}, Area: {area}')
        if name == 'distre':
            name = 'DISTRE'
        elif name == 'pcnnatt':
            name = 'PCNN-ATT'
        elif name == 'reside':
            name = 'RESIDE'
        elif name == 'intra-inter':
            name = 'INTRA-INTER'
        elif name == 'recon-sent':
            name = 'RECON-SENT'

        plt.plot(rec, prec, label=name, lw=1, marker=marker[i], color=color[i], markevery=0.2, ms=6)

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc="upper right", prop={'size': 12})

    plot_path = f'../plots/pr_curves_{dataset}'
    fig.savefig(plot_path + '.png', bbox_inches='tight')
    fig.savefig(plot_path + '.pdf', bbox_inches='tight')
    print('Precision-Recall plot saved at: {}'.format(plot_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['nyt10', 'nyt10_570k', 'wiki_distant'])
    args = parser.parse_args()
    plot_pr(args.dataset)
