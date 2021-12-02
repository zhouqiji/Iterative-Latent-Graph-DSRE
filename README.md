# RE-Constrained-Latent-Graph
Code for the paper: Relation Extraction by Constrained Latent Graph

## Prerequisites
We follow the pre-processing processes with [dsre-vae](https://github.com/fenchri/dsre-vae). 
### Environment
```bash
conda create -n CLG python=3.9
conda activate CLG
pip install -r requirements.txt
```


### Pretrained word embeddings
```bash
mkdir embeds
cd embeds 
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

### Dataset Pre-processing
```bath
cd data

# To get the data
python get_data.py --dataset [nyt10, nyt10_570k, wiki_distant]      # nyt10 before nyt10_570k for sharing file

# Pre-process the data
python preprocess.py --max_sent_len 50 --lowercase --max_bag_size 500 --dataset [nyt10, nyt10_570k, wiki_distant]

# To get the KB
TBD
```