# -*- coding: utf-8 -*-
"""
@object: Twitter
@task: Main function of recursive NN (4 classes)
@author: majing
@structure: Top-Down recursive Neural Networks
@variable: Nepoch, lr, obj, fold
@time: Jan 24, 2018
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import BiD_RvNN
import math

import BU_loadData
import TD_loadData

import numpy as np
import TD_RvNN
import time
import random
import torch.optim as optim
import datetime
from evaluate import *

obj = "Twitter15" # choose dataset, you can choose either "Twitter15" or "Twitter16"
fold = "3" # fold index, choose from 0-4
tag = ""
vocabulary_size = 5000
hidden_dim = 100
Nclass = 4
Nepoch = 600
lr = 0.005

unit="BU_RvNN-"+obj+str(fold)+'-vol.'+str(vocabulary_size)+tag

treePath = '../resource/data.BU_RvNN.vol_'+str(vocabulary_size)+tag+'.txt'

trainPath = "../nfold/RNNtrainSet_"+obj+str(fold)+"_tree.txt"
testPath = "../nfold/RNNtestSet_"+obj+str(fold)+"_tree.txt"
labelPath = "../resource/"+obj+"_label_All.txt"

##################################### MAIN ####################################        
## 1. load tree & word & index & label
BU_tree_train, BU_word_train, BU_index_train, BU_y_train, BU_tree_test, BU_word_test, BU_index_test, BU_y_test = BU_loadData.loadData(treePath,labelPath,trainPath,testPath)
TD_tree_train, TD_word_train, TD_index_train, TD_leaf_idxs_train, TD_y_train, TD_tree_test, TD_word_test, TD_index_test, TD_leaf_idxs_test, TD_y_test = TD_loadData.loadData(treePath,labelPath,trainPath,testPath)
assert (BU_y_train == TD_y_train)
assert (BU_y_test == TD_y_test)
print "load data completed"

## 2. ini RNN model
t0 = time.time()
model = BiD_RvNN.RvNN(vocabulary_size, hidden_dim, Nclass)
t1 = time.time()
print 'Recursive model established,', (t1-t0)/60

## 3. looping SGD
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
losses_5, losses = [], []
num_examples_seen = 0
indexs = [i for i in range(len(TD_y_train))]
for epoch in range(Nepoch):
    ## one SGD
    random.shuffle(indexs)
    for i in indexs:
        pred_y, loss = model.forward(TD_word_train[i], TD_index_train[i], TD_tree_train[i], TD_leaf_idxs_train[i], BU_word_train[i], BU_index_train[i], BU_tree_train[i], BU_y_train[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
        num_examples_seen += 1
        print("epoch=%d: idx=%d, loss=%f" % (epoch, i, np.mean(losses)))
        if i == indexs[10]:
            break
        ## cal loss & evaluate
    if epoch % 1 == 0:
        losses_5.append((num_examples_seen, np.mean(losses)))
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, np.mean(losses)))
        sys.stdout.flush()
        prediction = []
        for j in range(len(TD_y_test)):
            prediction.append(
                model.predict_up(TD_word_test[i], TD_index_test[i], TD_tree_test[i], TD_leaf_idxs_test[i], BU_word_test[i], BU_index_test[i], BU_tree_test[i], BU_y_test[i]).data.tolist())
        print("predictions:", prediction)
        res = evaluation_4class(prediction, TD_y_test)
        print('results:', res)
        sys.stdout.flush()
        ## Adjust the learning rate if loss increases
        if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
            lr = lr * 0.5
            print("Setting learning rate to %f" % lr)
            sys.stdout.flush()
    sys.stdout.flush()
    losses = []
model.SaveModels('ModelStorage/Initial.MIXmodel')
