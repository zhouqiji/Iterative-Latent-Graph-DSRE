# RE-Constrained-Latent-Graph
Code for the paper: Relation Extraction by Constrained Latent Graph

## Prerequisites
We get and pre-process the data similar to [dsre-vae](https://github.com/fenchri/dsre-vae).
### Environment
```bash
conda create -n CLG python=3.8
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
```bash
cd data

# To get the data
python get_data.py --dataset [nyt10, nyt10_570k, wiki_distant]      # nyt10 before nyt10_570k for sharing file

# Pre-process the data
python preprocess.py --max_sent_len 50 [30 for wikidata] --lowercase --max_bag_size 500 --path LANG/[nyt10, nyt10_570k, wiki_distant]  --dataset [nyt10, nyt10_570k, wiki_distant]

```

### KB Pre-processing
To get the Knowledge Base prior, we need to get and train the KB embeddings.
```bash
# To get the KB
cd data/KB
python make_data.py --data [Wikidata, Freebase] \
                    --train_file PATH_TO_TRAIN_FILE
                    --val_file PATH_TO_VAL_FILE
                    --test_file PATH_TO_TEST_FILE
```

### Training KB embeddings
In order to train Knowledge Base embeddings, we will use the [DGL-KE](https://github.com/awslabs/dgl-ke) package.
The following script will train TransE entity and relation embeddings for Freebase and Wikidata.
```bash
cd data/KB
sh train_embeds.sh
```
Embeddings will be saved in the `Freebase/ckpts_64/` and `Wikidata/ckpts_64/` directories, respectively.
Collect priors for your own KB:
```bash
python calculate_priors.py --kg_embeds Freebase/ckpts_64/TransE_l2_Freebase_0/Freebase_TransE_l2_entity.npy \
                           --e_map Freebase/entities.tsv \  # [Or Wikidata\...]
                           --data ../LANG/nyt10/nyt10_train.txt # [Or nyt10_570k Or wiki_distant]\
                           --kg [Freebase, Wikidata]
```
These can be **directly downloaded** [here](https://drive.google.com/file/d/1rqXQ3uqI0n98S5j7gPYaXgf1hQELIi_E/view?usp=sharing).
(Thanks for Fenia Christopoulou‘s open source)
#TODO: Clean unnecessary config and code------------------------