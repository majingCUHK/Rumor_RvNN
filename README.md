# Paper of the source codes released:

Jing Ma, Wei Gao, Kam-Fai Wong. Rumor Detection on Twitter with Tree-structured Recursive Neural Networks. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018.

# Datasets:

The datasets used in the experiments were based on the two publicly available Twitter datasets released by Ma et al. (2017):

Jing Ma, Wei Gao, Kam-Fai Wong. Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning. ACL 2017.

In the 'resource' folder we provide the pre-processed data files used for our experiments. The raw datasets can be downloaded from https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0. For details about the datasets please contact Jing at: majing at se dot cuhk dot edu dot hk.

The datafile is in a tab-sepreted column format, where each row corresponds to a tweet. Consecutive columns correspond to the following pieces of information:

1: rootid -- a unique identifier describing the tree.

2: index-of-parent-node -- an index number of the parent tweet for the current tweet.

3: index-of-current-node -- an index number of the current tweet

parent-number -- the total number of the tree that the current tweet is part of;

text-length -- the maxium length of all the texts from the tree that the current tweet is part of;

list-of-index-and-counts -- the rest of the line contains space separated index-count pairs, where a index-count pair is in format of "index:count", E.g., "index1:count1 index2:count2" (extracted from the "text" field in the json format from Twitter)


# Dependencies:
Please install the following python libraries:

numpy version 1.11.2

theano version 0.8.2

# Reproduce the experimental results
Run script "model/Main_tree_gru.py" for bottom-up model or "model/Main_tree_gru_u2b_pool.py" for up-down model.

Alternatively, you can change the "obj" parameter and "fold" parameter to set the dataset and each fold.

#If you find this code useful, please let us know and cite our paper.
