import os
import argparse
import logging
import wget

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
        id_file = wget.download(rel_id_url, out=_data_dir)



    elif data_name == 'nyt10_570k':
        pass
    else:
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['nyt10', 'nyt10_570k', 'wiki_distant'])
    args = parser.parse_args()

    if not os.path.exists(LANG_PATH):
        os.makedirs(LANG_PATH)


    download_data(args.dataset)
    #clean_data(args.data)

