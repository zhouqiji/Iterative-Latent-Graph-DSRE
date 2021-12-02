# RE-Constrained-Latent-Graph
Code for the paper: Relation Extraction by Constrained Latent Graph

## Prerequisites
We get and pre-process the data similar to [dsre-vae](https://github.com/fenchri/dsre-vae).
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
python preprocess.py --max_sent_len 50 --lowercase --max_bag_size 500 --path LANG/[nyt10, nyt10_570k, wiki_distant]  --dataset [nyt10, nyt10_570k, wiki_distant]

```

### KB Pre-processing
To get the Knowledge Base prior, we need to get and train the KB embeddings.
```bath
cd data

# To get the KB

```