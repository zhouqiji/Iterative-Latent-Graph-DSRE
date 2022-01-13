import os, sys, re
import argparse
import logging
import wget
import json
import zipfile
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# raw data path
LANG_PATH = './LANG/'


def download_data(data_name):
    _data_dir = os.path.join(LANG_PATH, data_name)

    if not os.path.exists(_data_dir):
        os.makedirs(_data_dir)

    # Download the data
    logging.info("Downloading {0} dataset ...".format(data_name))
    if data_name == 'nyt10':
        train_url = 'https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_train.txt'
        test_url = 'https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_test.txt'
        rel_id_url = 'https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_rel2id.json'

        train_file = wget.download(train_url, out=_data_dir)
        test_file = wget.download(test_url, out=_data_dir)
        # id_file = wget.download(rel_id_url, out=_data_dir)

        train_file_full = _data_dir + '/nyt10_train_full.txt'

        os.rename(train_file, train_file_full)

        id_file = './nyt10_rel2id.json'
        return _data_dir, train_file_full, train_file, id_file

    elif data_name == 'nyt10_570k':
        data_url = 'https://github.com/ZhixiuYe/Intra-Bag-and-Inter-Bag-Attentions/raw/master/NYT_data/NYT_data.zip'

        data_file = wget.download(data_url, out=_data_dir)
        with zipfile.ZipFile(data_file, 'r') as zip_ref:
            zip_ref.extractall(_data_dir)

        train_raw_file = _data_dir + '/train.txt'
        train_file_full = _data_dir + '/nyt10_570k_train_full.txt'
        train_file = _data_dir + '/nyt10_570k_train.txt'

        # fix_format
        def identify_argument(text, arg1):
            # find argument offsets
            if re.search(r'\b' + re.escape(arg1) + r'\b', text):
                start_offset_a = re.search(r'\b' + re.escape(arg1) + r'\b', text).start()

            elif re.search(re.escape(arg1) + r'\b', text):
                start_offset_a = re.search(re.escape(arg1) + r'\b', text).start()

            elif re.search(r'\b' + re.escape(arg1), text):
                start_offset_a = re.search(r'\b' + re.escape(arg1), text).start()

            else:
                assert False, 'Cannot find word == {}\n{}'.format(arg1, text)

            end_offset_a = start_offset_a + len(arg1)

            assert text[start_offset_a:end_offset_a] == arg1, \
                '{}\n{} <> Arg: {}'.format(text,
                                           text[start_offset_a:end_offset_a], arg1)

            return start_offset_a, end_offset_a

        with open(train_raw_file, 'r') as infile, open(train_file_full, 'w') as outfile:
            for line in tqdm(infile):
                line = line.rstrip().split('\t')
                arg1_id = line[0]
                arg2_id = line[1]
                name1 = line[2]
                name2 = line[3]
                relation = line[4]
                sentence = line[5].replace(' ###END###', '')

                sentence = sentence.replace(name1, name1.replace('_', ' '))
                sentence = sentence.replace(name2, name2.replace('_', ' '))
                name1 = name1.replace('_', ' ')
                name2 = name2.replace('_', ' ')

                offsets1 = identify_argument(sentence, name1)
                offsets2 = identify_argument(sentence, name2)

                out_dict = {'text': sentence,
                            'relation': relation,
                            'h': {'name': name1, 'id': arg1_id, 'pos': list(offsets1)},
                            't': {'name': name2, 'id': arg2_id, 'pos': list(offsets2)}}
                outfile.write(json.dumps(out_dict) + '\n')
        id_file = './nyt10_rel2id.json'
        return _data_dir, train_file_full, train_file, id_file

    else:
        # Download the wikidistant data
        test_url = 'https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki_distant/wiki_distant_test.txt'
        train_url = 'https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki_distant/wiki_distant_train.txt'
        val_url = 'https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki_distant/wiki_distant_val.txt'
        rel_id_url = 'https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki_distant/wiki_distant_rel2id.json'

        train_file = wget.download(train_url, out=_data_dir)
        test_file = wget.download(test_url, out=_data_dir)
        val_file = wget.download(val_url, out=_data_dir)
        id_file = wget.download(rel_id_url, out=_data_dir)


def clean_data(data_name, data):
    # unpack data
    _data_dir, train_file_full, train_file, id_file = data
    val_file = ''
    if data_name == 'nyt10':
        val_file = _data_dir + '/nyt10_val.txt'
    elif data_name == 'nyt10_570k':
        val_file = _data_dir + '/nyt10_570k_val.txt'
    else:
        pass

    # Clean nyt data
    with open(id_file, 'r') as infile:
        relations = json.load(infile)
        relations = list(relations.keys())

    train_bags = {}
    instances = 0

    with open(train_file_full, 'r') as infile:
        for line in infile:
            line = json.loads(line)
            instances += 1

            tmp = [line['text'], line['h']['pos'], line['t']['pos'], line['h']['name'], line['t']['name']]
            triple = (line['h']['id'], line['relation'], line['t']['id'])

            if line['relation'] not in relations:
                triple = (line['h']['id'], 'NA', line['t']['id'])

            if triple not in train_bags:
                train_bags[triple] = [tmp]
            else:
                train_bags[triple] += [tmp]

    print('Train instances:  {}'.format(instances))
    print('TRAIN bags:       {}'.format(len(train_bags)))
    print('Unique labels:    {}'.format(len(list(set(relations)))))

    counts = {}
    for r in relations:
        counts[r] = 0
    for b in train_bags:
        counts[b[1]] += 1

    exclude_items = {}
    keep_items = {}
    new_relations = []
    for b in train_bags:
        if counts[b[1]] == 1:
            exclude_items[b] = train_bags[b]
        else:
            keep_items[b] = train_bags[b]
            new_relations += [b[1]]

    print('TRAIN (relations > 2): {}'.format(len(keep_items)))
    print('TRAIN labels (relations > 2): {}'.format(len(new_relations)))

    print('Splitting training set into train and validation (90/10) ...')

    X_train, X_val, y_train, y_val = train_test_split(np.arange(len(keep_items)),
                                                      new_relations,
                                                      test_size=0.10,
                                                      random_state=42,
                                                      stratify=new_relations)

    print("Storing files ... ")

    with open(train_file, 'w') as outfile:
        for i, ins in enumerate(keep_items):
            if i in X_train:
                for tmp in keep_items[ins]:
                    line = {'text': tmp[0], 'h': {'pos': tmp[1], 'id': ins[0], 'name': tmp[3]},
                            't': {'pos': tmp[2], 'id': ins[2], 'name': tmp[4]}, 'relation': ins[1]}
                    outfile.write(json.dumps(line))
                    outfile.write('\n')

        # Add unis
        for ins in exclude_items:
            for tmp in exclude_items[ins]:
                line = {'text': tmp[0], 'h': {'pos': tmp[1], 'id': ins[0], 'name': tmp[3]},
                        't': {'pos': tmp[2], 'id': ins[2], 'name': tmp[4]}, 'relation': ins[1]}
                outfile.write(json.dumps(line))
                outfile.write('\n')

    with open(val_file, 'w') as outfile:
        for i, ins in enumerate(keep_items):
            if i in X_val:
                for tmp in keep_items[ins]:
                    line = {'text': tmp[0], 'h': {'pos': tmp[1], 'id': ins[0], 'name': tmp[3]},
                            't': {'pos': tmp[2], 'id': ins[2], 'name': tmp[4]}, 'relation': ins[1]}
                    outfile.write(json.dumps(line))
                    outfile.write('\n')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['nyt10', 'nyt10_570k', 'wiki_distant'])
    args = parser.parse_args()

    if not os.path.exists(LANG_PATH):
        os.makedirs(LANG_PATH)

    data = download_data(args.dataset)
    if args.dataset.startswith('nyt'):
        clean_data(args.dataset, data)
