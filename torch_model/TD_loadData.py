# -*- coding: utf-8 -*-
import TD_RvNN

################################### tools #####################################
def str2matrix(Str, MaxL):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    l = 0
    for pair in Str.split(' '):
        wordFreq.append(float(pair.split(':')[1]))
        wordIndex.append(int(pair.split(':')[0]))
        l += 1
    ladd = [0 for i in range(MaxL - l)]
    wordFreq += ladd
    wordIndex += ladd
    return wordFreq, wordIndex


def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if label in labelset_nonR:
        y_train = [1, 0, 0, 0]
        l1 += 1
    if label in labelset_f:
        y_train = [0, 1, 0, 0]
        l2 += 1
    if label in labelset_t:
        y_train = [0, 0, 1, 0]
        l3 += 1
    if label in labelset_u:
        y_train = [0, 0, 0, 1]
        l4 += 1
    return y_train, l1, l2, l3, l4


def constructTree(tree):
    ## tree: {index1:{'parent':, 'maxL':, 'vec':}
    ## 1. ini tree node
    index2node = {}
    for i in tree:
        node = TD_RvNN.Node_tweet(idx=i)
        index2node[i] = node
    ## 2. construct tree
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'], tree[j]['maxL'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        # nodeC.time = tree[j]['post_t']
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
    ## 3. convert tree to DNN input
    parent_num = tree[j]['parent_num']
    ini_x, ini_index = str2matrix("0:0", tree[j]['maxL'])
    x_word, x_index, tree, leaf_idxs = TD_RvNN.gen_nn_inputs(root, ini_x)
    return x_word, x_index, tree, leaf_idxs


################################# loas data ###################################
def loadData(treePath, labelPath, trainPath, testPath):
    print("loading tree label",)
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        labelDic[eid] = label.lower()
    print(len(labelDic))

    print("reading tree"),  ## X
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        parent_num, maxL = int(line.split('\t')[3]), int(line.split('\t')[4])
        Vec = line.split('\t')[5]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'parent_num': parent_num, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))

    print("loading train set",)
    tree_train, word_train, index_train, y_train, leaf_idxs_train, c = [], [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    for eid in open(trainPath):
        # if c > 8: break
        eid = eid.rstrip()
        if not labelDic.__contains__(eid): continue
        if not treeDic.__contains__(eid): continue
        if len(treeDic[eid]) < 2:
            continue
        ## 1. load label
        label = labelDic[eid]
        y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
        y_train.append(y)
        ## 2. construct tree
        x_word, x_index, tree, leaf_idxs = constructTree(treeDic[eid])
        tree_train.append(tree)
        word_train.append(x_word)
        index_train.append(x_index)
        leaf_idxs_train.append(leaf_idxs)
        c += 1
    print(l1, l2, l3, l4)

    print("loading test set",)
    tree_test, word_test, index_test, leaf_idxs_test, y_test, c = [], [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    for eid in open(testPath):
        # if c > 4: break
        eid = eid.rstrip()
        if not labelDic.__contains__(eid): continue
        if not treeDic.__contains__(eid): continue
        if len(treeDic[eid]) < 2:
            continue
            ## 1. load label
        label = labelDic[eid]
        y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        ## 2. construct tree
        x_word, x_index, tree, leaf_idxs = constructTree(treeDic[eid])
        tree_test.append(tree)
        word_test.append(x_word)
        index_test.append(x_index)
        leaf_idxs_test.append(leaf_idxs)
        c += 1
    print(l1, l2, l3, l4)
    print("train no:", len(tree_train), len(word_train), len(index_train), len(leaf_idxs_train), len(y_train))
    print("test no:", len(tree_test), len(word_test), len(index_test), len(leaf_idxs_test), len(y_test))
    print("dim1 for 0:", len(tree_train[0]), len(word_train[0]), len(index_train[0]))
    print("case 0:", tree_train[0][0], word_train[0][0], index_train[0][0], leaf_idxs_train[0])
    return tree_train, word_train, index_train, leaf_idxs_train, y_train, tree_test, word_test, index_test, leaf_idxs_test, y_test

