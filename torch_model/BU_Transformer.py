#-*- encoding: utf8

__doc__ = """Tree GRU aka Recursive Neural Networks."""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import Transformer_Utils
import math
import copy

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

################################# generate tree structure ##############################
def gen_nn_inputs(root_node, max_degree=None, only_leaves_have_vals=True, with_labels=False):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    _clear_indices(root_node)
    X_word, X_index = _get_leaf_vals(root_node)
    tree, internal_word, internal_index = _get_tree_traversal(root_node, len(X_word), max_degree)
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)
    '''if with_labels:
        labels = leaf_labels + internal_labels
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return (np.array(x, dtype='int32'),
                np.array(tree, dtype='int32'),
                np.array(labels, dtype=theano.config.floatX),
                np.array(labels_exist, dtype=theano.config.floatX))'''
    ##### debug here #####
    '''ls = []
    for x in X_word:
        l = len(x)
        if not l in ls: ls.append(l)
    print ls'''
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'))


def _clear_indices(root_node):
    root_node.idx = None
    [_clear_indices(child) for child in root_node.children if child]


def _get_leaf_vals(root_node):
    """Get leaf values in deep-to-shallow, left-to-right order."""
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if not node.children:
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    X_word = []
    X_index = []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        X_word.append(leaf.word)
        X_index.append(leaf.index)
    return X_word, X_index


def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer

    tree = []
    internal_word = []
    internal_index = []
    idx = start_idx
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue

            child_idxs = [(child.idx if child else -1)
                          for child in node.children]  ## idx of child node
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_word.append(node.word if node.word is not None else -1)
            internal_index.append(node.index if node.index is not None else -1)
            idx += 1

    return tree, internal_word, internal_index

################################ tree rnn class ######################################
#implentation of three models:
# 1. attention+GRU  --xh
# 2. multi_head attention + gru
# 3. multi_head attention + full connect


class AttentionGRU(nn.Module):
    def __init__(self, word_dim, hidden_dim=5, Nclass=4,
                 degree=2, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):
        super(AttentionGRU, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree  # 这里比较奇怪的是，在创建模型的时候是没有对degree进行赋值的
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        self.E_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.word_dim]), requires_grad=True)
        self.W_z_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_z_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_z_bu = nn.parameter.Parameter(self.init_vector([1, self.hidden_dim]), requires_grad=True)
        self.W_r_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_r_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_r_bu = nn.parameter.Parameter(self.init_vector([1, self.hidden_dim]), requires_grad=True)
        self.W_h_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_h_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_h_bu = nn.parameter.Parameter(self.init_vector([1, self.hidden_dim]), requires_grad=True)
        self.W_out_bu = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad=True)
        self.b_out_bu = nn.parameter.Parameter(self.init_vector(self.Nclass), requires_grad=True)

        # self.WQ = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        # self.WK = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        # self.WV = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.W_attn = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.norm = nn.LayerNorm([1,self.hidden_dim])
        self.drop = nn.Dropout(0.1)

    def forward(self, x_word, x_index, tree):
        final_state = self.compute_tree_states(x_word, x_index, tree)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)

    def recursive_unit(self, parent_xe, child_h, child_xe):
        # #self attention
        # if child_xe.size(0) == 1:
        #     h_tilde = child_h.mm(self.WV)
        # else:
        #     query = parent_xe.mm(self.WQ)
        #     key = child_h.mm(self.WK)
        #     val = child_h.mm(self.WV)
        #     attention = F.softmax( (query/np.sqrt(self.hidden_dim*1.0)).mm(key.t()))
        #     h_tilde = attention.mm(val)
        #attention
        attn = F.softmax(F.sigmoid(parent_xe.mm(self.W_attn.mm(child_h.t()))))
        h_tilde = attn.mm(child_h)
        #gru
        self.norm(h_tilde)
        z_bu = F.sigmoid(parent_xe.mm(self.W_z_bu.t()) + h_tilde.mm(self.U_z_bu.t()) + self.b_z_bu)
        r_bu = F.sigmoid(parent_xe.mm(self.W_r_bu.t()) + h_tilde.mm(self.U_r_bu.t()) + self.b_r_bu)
        c = F.tanh(parent_xe.mm(self.W_h_bu.t()) + (r_bu*h_tilde).mm(self.U_h_bu.t()) + self.b_h_bu)
        h_bu = z_bu * h_tilde + (1 - z_bu) * self.drop(c)
        return h_bu

    def Word2Vec(self, word, index):
        vec = torch.tensor([word]).mm(self.E_bu[:, index].t())
        return vec

    def Words2Vecs(self, words, indexs):
        tmp = torch.tensor([])
        for i in range(len(words)):
            tmp = torch.cat((tmp, self.Word2Vec(words[i], indexs[i])), dim=0)
        return tmp

    def compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes -num_parents
        leaf_h = list(map(
                            lambda x_idxs: self.recursive_unit( self.Word2Vec(x_idxs[0], x_idxs[1]), torch.zeros([self.degree, self.hidden_dim]), torch.zeros([self.degree, self.hidden_dim])).tolist()[0],
                                zip(x_word[:num_leaves], x_index[:num_leaves])
                        )
                    )

        init_node_h = torch.tensor(leaf_h)

        def _recurrence(x_word, indexs, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            if len(tree[child_exists]) == 1:
                child_xes = self.Word2Vec(x_word[ tree[child_exists][0] ], indexs[ tree[child_exists][0] ])
                child_h = node_h[tree[child_exists][0]].unsqueeze(0)
                # child_h = node_h[tree[child_exists][0]].unsqueeze(0)
                # print("************debug1:", child_h.size())
            else:
                child_xes = self.Words2Vecs(x_word[ tree[child_exists] ], indexs[ tree[child_exists] ])
                child_h = node_h[tree[child_exists]]
            parent_xe = self.Word2Vec(x_word[num_leaves+idx], indexs[num_leaves+idx])
            parent_h = self.recursive_unit(parent_xe, child_h, child_xes)
            node_h = torch.cat((node_h, parent_h.view(1, -1)), 0)
            return node_h, parent_h

        node_h = init_node_h
        for idx, thislayer in enumerate(tree):
            node_h, parent_h = _recurrence(x_word, x_index, thislayer, idx, node_h)
        return node_h[num_leaves:].max(dim=0)[0]

    def predAndLoss(self, final_state, ylabel):
        pred = F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)
        loss = (torch.tensor(ylabel, dtype=torch.float)-pred).pow(2).sum()
        return pred, loss

    def init_vector(self, shape):
        return torch.zeros(shape)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, x_word, x_index, x_tree):
        final_state = self.compute_tree_states(x_word, x_index, x_tree)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)

class MultiAttentionGRU(nn.Module):
    def __init__(self, word_dim, hidden_dim=64, Nclass=4,
                 degree=2, momentum=0.9, multi_head=5,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):
        super(MultiAttentionGRU, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree  # 这里比较奇怪的是，在创建模型的时候是没有对degree进行赋值的
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        self.E_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.word_dim]), requires_grad=True)
        self.W_z_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_z_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_z_bu = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_r_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_r_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_r_bu = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_h_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_h_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_h_bu = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_out_bu = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad=True)
        self.b_out_bu = nn.parameter.Parameter(self.init_vector([self.Nclass]), requires_grad=True)

        self.multi_head = multi_head

        self.WQ = nn.parameter.Parameter(self.init_matrix([self.multi_head, self.hidden_dim, self.hidden_dim]))
        self.WK = nn.parameter.Parameter(self.init_matrix([self.multi_head, self.hidden_dim, self.hidden_dim]))
        self.WV = nn.parameter.Parameter(self.init_matrix([self.multi_head, self.hidden_dim, self.hidden_dim]))
        self.WO = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim*self.multi_head]))

        self.LM = nn.LayerNorm(self.hidden_dim)
        self.Drop = nn.Dropout(0.5)
    def forward(self, x_word, x_index, tree, y):
        final_state = self.compute_tree_states(x_word, x_index, tree)
        pred, loss = self.predAndLoss(final_state, y)
        return loss

    def Word2Vec(self, word, index):
        vec = self.E_bu[:, index].mul(torch.tensor(word)).sum(dim=1)
        return vec

    def Words2Vecs(self, words, indexs):
        words = torch.tensor(words)
        tmp = torch.tensor([])
        for i in range(words.size()[0]):
            tmp = torch.cat((tmp, self.Word2Vec(words[i], indexs[i])), dim=0)
        vec = tmp.view(words.size()[0], -1)
        return vec

    def AttentionedSumOfChilds(self, parent_xe, child_h, child_xe):
        query = parent_xe.mul(self.WQ[0]).sum(dim=1)
        key = torch.tensor(child_xe).mm(self.WK[0])
        # val = child_h.mm(self.WV[0])
        attention = F.softmax((query / np.sqrt(self.hidden_dim * 1.0)).mul(key).sum(dim=1))
        h_tilde = attention.mul(child_h.t()).sum(dim=1)
        for i in range(1, self.multi_head):
            query = parent_xe.mul(self.WQ[i]).sum(dim=1)
            key = child_xe.mm(self.WK[i])
            # val = child_h.mm(self.WV[i])
            attention = F.softmax((query * np.sqrt(self.hidden_dim * 1.0)).mul(key).sum(dim=1))
            tmp = attention.mul(child_h.t()).sum(dim=1)
            h_tilde = torch.cat((h_tilde, tmp), dim=0)
        return h_tilde


    def recursive_unit(self, parent_xe, child_h, child_xe):
        #attention
        if child_xe.dim() ==1:
            child_h_XL = torch.cat(tuple((child_h for i in range(self.multi_head))), 0)
            h_tilde = self.WO.mul(child_h_XL).sum(dim=1)
        else:
            child_h_XL = self.AttentionedSumOfChilds(parent_xe, child_h, child_xe)
            h_tilde = self.WO.mul(child_h_XL).sum(dim=1)

        h_tilde = self.LM(h_tilde)
        #gru
        z_bu = F.sigmoid(self.W_z_bu.mul(parent_xe).sum(dim=1) + self.U_z_bu.mul(h_tilde).sum(dim=1) + self.b_z_bu)
        r_bu = F.sigmoid(self.W_r_bu.mul(parent_xe).sum(dim=1) + self.U_r_bu.mul(h_tilde).sum(dim=1) + self.b_r_bu)
        c = F.tanh(self.W_h_bu.mul(parent_xe).sum(dim=1) + self.U_h_bu.mul(h_tilde * r_bu).sum(dim=1) + self.b_h_bu)
        h_bu = z_bu * h_tilde + (1 - z_bu) * self.Drop(c)
        return h_bu

    def compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes -num_parents
        leaf_h = list(map(
                            lambda x_idxs: self.recursive_unit( self.Word2Vec(x_idxs[0], x_idxs[1]), torch.zeros([self.degree, self.hidden_dim]), torch.zeros([self.degree, self.hidden_dim])).tolist(),
                                zip(x_word[:num_leaves], x_index[:num_leaves])
                        )
                    )

        init_node_h = torch.tensor(leaf_h)

        def _recurrence(x_word, indexs, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            if len(tree[child_exists]) == 1:
                child_xes = self.Word2Vec(x_word[ tree[child_exists][0] ], indexs[ tree[child_exists][0] ])
                child_h = node_h[tree[child_exists][0]]
            else:
                child_xes = self.Words2Vecs(x_word[ tree[child_exists] ], indexs[ tree[child_exists] ])
                child_h = node_h[tree[child_exists]]
            parent_xe = self.Word2Vec(x_word[num_leaves+idx], indexs[num_leaves+idx])
            parent_h = self.recursive_unit(parent_xe, child_h, child_xes)
            node_h = torch.cat((node_h, parent_h.view(1, -1)), 0)
            return node_h, parent_h

        node_h = init_node_h
        root_state = []
        for idx, thislayer in enumerate(tree):
            node_h, parent_h = _recurrence(x_word, x_index, thislayer, idx, node_h)
            if idx == num_parents-1:
                root_state = parent_h
        return root_state

    def predAndLoss(self, final_state, ylabel):
        pred = F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)
        loss = (torch.tensor(ylabel, dtype=torch.float)-pred).pow(2).sum()
        return pred, loss

    def init_vector(self, shape):
        return torch.zeros(shape)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, x_word, x_index, x_tree):
        final_state = self.compute_tree_states(x_word, x_index, x_tree)
        final_state = self.Drop(final_state)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)

class TransformerEncoder(nn.Module): # performance: twitter15 (73.81%) twitter16(74.49%)
    def __init__(self, word_dim, hidden_dim=64, Nclass=4,
                 degree=2, momentum=0.9, multi_head=8,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):
        super(TransformerEncoder, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree  # 这里比较奇怪的是，在创建模型的时候是没有对degree进行赋值的
        self.momentum = momentum
        self.irregular_tree = irregular_tree
        self.multi_head = multi_head

        self.E_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.word_dim]), requires_grad=True)
        self.W_out_bu = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad=True)
        self.b_out_bu = nn.parameter.Parameter(self.init_vector([self.Nclass]), requires_grad=True)

        c = copy.deepcopy
        attn = Transformer_Utils.MultiHeadedAttention(self.multi_head, self.hidden_dim)
        ffw = Transformer_Utils.PositionwiseFeedForward(self.hidden_dim, self.hidden_dim*2, 0.1)

        self.encoder = Transformer_Utils.Encoder(Transformer_Utils.EncoderLayer(self.hidden_dim, c(attn), c(ffw), 0.1), 2)
        self.decoder = Transformer_Utils.Decoder(Transformer_Utils.DecoderLayer(self.hidden_dim, c(attn), c(attn), c(ffw), 0.1), 2)


    def forward(self, x_word, x_index, tree):
        final_state = self.compute_tree_states(x_word, x_index, tree)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) + self.b_out_bu)

    def compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes - num_parents
        leaf_h = list(map(
                            lambda x_idxs: self.E_bu[:, x_idxs[1]].mul(torch.from_numpy(x_idxs[0])).sum(dim=1).tolist(),
                                zip(x_word[:num_leaves], x_index[:num_leaves])
                        )
                    )

        init_node_h = torch.tensor(leaf_h)

        def _recurrence(x_word, x_index, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            child_h = node_h[ tree[child_exists] ]
            memory = self.encoder(child_h, mask=None)
            parent_xe = self.E_bu[:, x_index].mul(torch.tensor(x_word)).sum(dim=1)
            # sys.exit(0)
            parent_h = self.decoder(parent_xe, memory, None, None)
            node_h = torch.cat((node_h, parent_h.view(1, -1)), 0)
            return node_h, parent_h

        node_h = init_node_h
        root_state = []
        for idx, (words, indexs, thislayer) in enumerate(zip(x_word[num_leaves:], x_index[num_leaves:], tree)):
            node_h, parent_h = _recurrence(words, indexs, thislayer, idx, node_h)
            if idx == num_parents-1:
                root_state = parent_h
        return root_state

    def predAndLoss(self, final_state, ylabel):
        pred = F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)
        loss = (torch.tensor(ylabel, dtype=torch.float)-pred).pow(2).sum()
        return pred, loss

    def init_vector(self, shape):
        return torch.zeros(shape)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    # def predict_up(self, x_word, x_index, x_tree):
    #     final_state = self.compute_tree_states(x_word, x_index, x_tree)
    #     return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)

class StarTransformer(nn.Module): # performance: twitter15 (73.81%) twitter16(74.49%)
    def __init__(self, word_dim, hidden_dim=64, Nclass=4,
                 degree=2, momentum=0.9, multi_head=8,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):
        super(StarTransformer, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree  # 这里比较奇怪的是，在创建模型的时候是没有对degree进行赋值的
        self.momentum = momentum
        self.irregular_tree = irregular_tree
        self.multi_head = multi_head

        self.E_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.word_dim]), requires_grad=True)
        self.W_out_bu = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad=True)
        self.b_out_bu = nn.parameter.Parameter(self.init_vector([self.Nclass]), requires_grad=True)
        self.Drop = nn.Dropout(0.1)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        # query = [head, nbatch, d_k] , key = [head, nbatch, d_k], scores = [head, nbatch, n_batch], x= [head, nbatch, nbatch][head, nbatch, d_k]
        d_k = query.size(-1)  # the dim of the query
        head = query.size(1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        # if dropout is not None: # drop out the attention is confusing
        #     p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, x_word, x_index, tree):
        final_state = self.compute_tree_states(x_word, x_index, tree)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) + self.b_out_bu)

    def compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes - num_parents
        leaf_h = list(map(
                            lambda x_idxs: self.E_bu[:, x_idxs[1]].mul(torch.from_numpy(x_idxs[0])).sum(dim=1).tolist(),
                                zip(x_word[:num_leaves], x_index[:num_leaves])
                        )
                    )

        init_node_h = torch.tensor(leaf_h)

        def _recurrence(x_word, x_index, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            child_h = node_h[ tree[child_exists] ]
            memory = self.transformer(child_h, 5)
            parent_xe = self.E_bu[:, x_index].mul(torch.tensor(x_word)).sum(dim=1)
            z_bu = F.sigmoid(self.W_z_bu.mul(parent_xe).sum(dim=1) + self.U_z_bu.mul(memory).sum(dim=1) + self.b_z_bu)
            r_bu = F.sigmoid(self.W_r_bu.mul(parent_xe).sum(dim=1) + self.U_r_bu.mul(memory).sum(dim=1) + self.b_r_bu)
            c = F.tanh(self.W_h_bu.mul(parent_xe).sum(dim=1) + self.U_h_bu.mul(memory * r_bu).sum(dim=1) + self.b_h_bu)
            parent_h = z_bu * memory + (1 - z_bu) * self.Drop(c)
            node_h = torch.cat((node_h, parent_h.view(1, -1)), 0)
            return node_h, parent_h

        node_h = init_node_h
        for idx, (words, indexs, thislayer) in enumerate(zip(x_word[num_leaves:], x_index[num_leaves:], tree)):
            node_h, parent_h = _recurrence(words, indexs, thislayer, idx, node_h)
        return node_h[num_leaves:].max(dim=0)[0]

    def init_vector(self, shape):
        return torch.zeros(shape)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

class TransformerEncoderPoolV1(nn.Module): #twitter16(75.79%) ['acc:', 0.7579, 'Favg:', 0.7567, 0.6251, 0.3037, 'C1:', 0.8421, 0.9167, 0.44, 0.5946, 'C2:', 0.8211, 0.6889, 0.9118, 0.7848, 'C3:', 0.9684, 0.9333, 0.875, 0.9032, 'C4:', 0.8842, 0.6957, 0.8, 0.7442]
    def __init__(self, word_dim, hidden_dim=64, Nclass=4,
                 degree=2, momentum=0.9, multi_head=8,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):
        super(TransformerEncoderPoolV1, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree  # 这里比较奇怪的是，在创建模型的时候是没有对degree进行赋值的
        self.momentum = momentum
        self.irregular_tree = irregular_tree
        self.multi_head = multi_head

        self.E_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.word_dim]), requires_grad=True)
        self.W_out_bu = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad=True)
        self.b_out_bu = nn.parameter.Parameter(self.init_vector([self.Nclass]), requires_grad=True)

        c = copy.deepcopy
        attn = Transformer_Utils.MultiHeadedAttention(self.multi_head, self.hidden_dim)
        ffw = Transformer_Utils.PositionwiseFeedForward(self.hidden_dim, self.hidden_dim*2, 0.1)

        self.encoder = Transformer_Utils.Encoder(Transformer_Utils.EncoderLayer(self.hidden_dim, c(attn), c(ffw), 0.1), 2)
        self.decoder = Transformer_Utils.Decoder(Transformer_Utils.DecoderLayer(self.hidden_dim, c(attn), c(attn), c(ffw), 0.1), 2)


    def forward(self, x_word, x_index, tree):
        final_state = self.compute_tree_states(x_word, x_index, tree)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) + self.b_out_bu)

    def compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes - num_parents
        leaf_h = list(map(
                            lambda x_idxs: self.E_bu[:, x_idxs[1]].mul(torch.from_numpy(x_idxs[0])).sum(dim=1).tolist(),
                                zip(x_word[:num_leaves], x_index[:num_leaves])
                        )
                    )

        init_node_h = torch.tensor(leaf_h)

        def _recurrence(x_word, x_index, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            child_h = node_h[ tree[child_exists] ]
            memory = self.encoder(child_h, mask=None)
            parent_xe = self.E_bu[:, x_index].mul(torch.tensor(x_word)).sum(dim=1)
            # sys.exit(0)
            parent_h = self.decoder(parent_xe, memory, None, None)
            node_h = torch.cat((node_h, parent_h.view(1, -1)), 0)
            return node_h, parent_h

        node_h = init_node_h
        for idx, (words, indexs, thislayer) in enumerate(zip(x_word[num_leaves:], x_index[num_leaves:], tree)):
            node_h, parent_h = _recurrence(words, indexs, thislayer, idx, node_h)
        return node_h.max(dim=0)[0]

    def predAndLoss(self, final_state, ylabel):
        pred = F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)
        loss = (torch.tensor(ylabel, dtype=torch.float)-pred).pow(2).sum()
        return pred, loss

    def init_vector(self, shape):
        return torch.zeros(shape)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))


class TransformerEncoderPoolV2(nn.Module): #这个模型训练不出来，说明传播过程确实很重要
    def __init__(self, word_dim, hidden_dim=64, Nclass=4,
                 degree=2, momentum=0.9, multi_head=8,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):
        super(TransformerEncoderPoolV2, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree  # 这里比较奇怪的是，在创建模型的时候是没有对degree进行赋值的
        self.momentum = momentum
        self.irregular_tree = irregular_tree
        self.multi_head = multi_head

        self.E_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.word_dim]), requires_grad=True)
        self.W_out_bu = nn.parameter.Parameter(self.init_matrix([self.Nclass, 2*self.hidden_dim]), requires_grad=True)
        self.b_out_bu = nn.parameter.Parameter(self.init_vector([self.Nclass]), requires_grad=True)

        c = copy.deepcopy
        attn = Transformer_Utils.MultiHeadedAttention(self.multi_head, self.hidden_dim)
        ffw = Transformer_Utils.PositionwiseFeedForward(self.hidden_dim, self.hidden_dim*2, 0.1)

        self.encoder = Transformer_Utils.Encoder(Transformer_Utils.EncoderLayer(self.hidden_dim, c(attn), c(ffw), 0.1), 2)
        self.decoder = Transformer_Utils.Decoder(Transformer_Utils.DecoderLayer(self.hidden_dim, c(attn), c(attn), c(ffw), 0.1), 2)


    def forward(self, x_word, x_index, tree):
        final_state = self.compute_tree_states(x_word, x_index, tree)
        # print("w_shape:", self.W_out_bu.shape, "\n final_state_shape:", final_state.shape)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) + self.b_out_bu)

    def compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes - num_parents
        words_xe = torch.tensor(list(map(
                            lambda x_idxs: self.E_bu[:, x_idxs[1]].mul(torch.from_numpy(x_idxs[0])).sum(dim=1).tolist(),
                                zip(x_word, x_index)
                        )
                    )
        )

        def _recurrence(x_word, x_index, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            child_h = node_h[ tree[child_exists] ]
            memory = self.encoder(child_h, mask=None)
            parent_xe = self.E_bu[:, x_index].mul(torch.tensor(x_word)).sum(dim=1)
            # sys.exit(0)
            parent_h = self.decoder(parent_xe, memory, None, None)
            return parent_h

        node_h = torch.tensor(list(map(
                    lambda params: _recurrence(params[0], params[1], params[2], 0, words_xe).tolist(), zip(x_word[num_leaves:], x_index[num_leaves:], tree)
                )
            )
        )
        return torch.cat((node_h.max(dim=0)[0], node_h.min(dim=0)[0]), dim=1)

    def predAndLoss(self, final_state, ylabel):
        pred = F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)
        loss = (torch.tensor(ylabel, dtype=torch.float)-pred).pow(2).sum()
        return pred, loss

    def init_vector(self, shape):
        return torch.zeros(shape)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))


class MultiAttentionFCN(nn.Module):
    def __init__(self, word_dim, hidden_dim=5, Nclass=4,
                 degree=2, momentum=0.9, multi_head=5,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):
        super(MultiAttentionFCN, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree  # 这里比较奇怪的是，在创建模型的时候是没有对degree进行赋值的
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        self.E_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.word_dim]), requires_grad=True)
        self.W_z_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_z_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_z_bu = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_r_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_r_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_r_bu = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_h_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_h_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_h_bu = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_out_bu = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad=True)
        self.b_out_bu = nn.parameter.Parameter(self.init_vector([self.Nclass]), requires_grad=True)

        self.multi_head = multi_head

        self.WQ = nn.parameter.Parameter(self.init_matrix([self.multi_head, self.hidden_dim, self.hidden_dim]))
        self.WK = nn.parameter.Parameter(self.init_matrix([self.multi_head, self.hidden_dim, self.hidden_dim]))
        self.WV = nn.parameter.Parameter(self.init_matrix([self.multi_head, self.hidden_dim, self.hidden_dim]))
        self.WO = [nn.parameter.Parameter(self.init_matrix([(self.multi_head-i-1)*self.hidden_dim, self.hidden_dim*(self.multi_head-i)])) for i in range(self.multi_head-1)]
    def forward(self, x_word, x_index, tree, y):
        final_state = self.compute_tree_states(x_word, x_index, tree)
        pred, loss = self.predAndLoss(final_state, y)
        return pred, loss

    def recursive_unit(self, parent_word, parent_index, child_h):
        #attention
        parent_xe = self.E_bu[:, parent_index].mul(parent_word).sum(dim=1)

        query = parent_xe.mul(self.WQ[0].t()).sum(dim=1)
        key = child_h.mm(self.WK[0])
        val = child_h.mm(self.WV[0])
        attention = F.softmax( (query/np.sqrt(self.hidden_dim*1.0)).mul(key).sum(dim=1) )
        h_tilde = attention.mul(val.t()).sum(dim=1)

        for i in range(1,self.multi_head):
            query = parent_xe.mul(self.WQ[i].t()).sum(dim=1)
            key = child_h.mm(self.WK[i])
            val = child_h.mm(self.WV[i])
            attention = F.softmax((query * np.sqrt(self.hidden_dim * 1.0)).mul(key).sum(dim=1))
            tmp = attention.mul(val.t()).sum(dim=1)
            h_tilde = torch.cat((h_tilde, tmp), dim=0)

        for i in range(self.multi_head-1):
            h_tilde = F.sigmoid(self.WO[i].mul(h_tilde).sum(dim=1))

        return h_tilde


    def compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes -num_parents
        leaf_h = list(map(
                            lambda x_idxs: self.recursive_unit(torch.tensor(x_idxs[0]).cuda(), x_idxs[1], torch.zeros([self.degree, self.hidden_dim]).cuda()).tolist(),
                                zip(x_word[:num_leaves], x_index[:num_leaves])
                        )
                    )

        init_node_h = torch.tensor(leaf_h).cuda()

        def _recurrence(x_word, x_index, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            child_h = node_h[ tree[child_exists] ]
            parent_h = self.recursive_unit(torch.tensor(x_word).cuda(), x_index, child_h)
            node_h = torch.cat((node_h, parent_h.view(1, -1)), 0)
            return node_h, parent_h

        node_h = init_node_h
        root_state = []
        for idx, (words, indexs, thislayer) in enumerate(zip(x_word[num_leaves:], x_index[num_leaves:], tree)):
            node_h, parent_h = _recurrence(words, indexs, thislayer, idx, node_h)
            if idx == num_parents-1:
                root_state = parent_h
        return root_state

    def predAndLoss(self, final_state, ylabel):
        pred = F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)
        loss = (torch.tensor(ylabel, dtype=torch.float).cuda()-pred).pow(2).sum()
        return pred, loss

    def init_vector(self, shape):
        return torch.zeros(shape)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, x_word, x_index, x_tree):
        final_state = self.compute_tree_states(x_word, x_index, x_tree)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)

class TreeLSTM_2ary(nn.Module):

    def __init__(self, hiddendim=50, Nclass=5, max_degree=2, worddim=300):
        super(TreeLSTM_2ary, self).__init__()
        self.hiddendim = hiddendim
        self.Nclass = Nclass
        self.degree = max_degree
        self.word_dim = worddim
        # parameters for the model
        self.W_i = nn.parameter.Parameter(self.init_matrix([self.hiddendim, self.word_dim]))
        self.U_i = nn.parameter.Parameter(self.init_matrix([self.degree, self.hiddendim, self.hiddendim]))
        self.b_i = nn.parameter.Parameter(self.init_vector([self.hiddendim]))

        self.W_f = nn.parameter.Parameter(self.init_matrix([self.hiddendim, self.word_dim]))
        self.U_f = nn.parameter.Parameter(
            self.init_matrix([self.degree, self.degree, self.hiddendim, self.hiddendim]))
        self.b_f = nn.parameter.Parameter(self.init_vector([self.hiddendim]))

        self.W_o = nn.parameter.Parameter(self.init_matrix([self.hiddendim, self.word_dim]))
        self.U_o = nn.parameter.Parameter(self.init_matrix([self.degree, self.hiddendim, self.hiddendim]))
        self.b_o = nn.parameter.Parameter(self.init_vector([self.hiddendim]))

        self.W_u = nn.parameter.Parameter(self.init_matrix([self.hiddendim, self.word_dim]))
        self.U_u = nn.parameter.Parameter(self.init_matrix([self.degree, self.hiddendim, self.hiddendim]))
        self.b_u = nn.parameter.Parameter(self.init_vector([self.hiddendim]))

        self.W_s = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hiddendim]))
        self.b_s = nn.parameter.Parameter(self.init_vector([self.Nclass]))

        self.drop = nn.Dropout(p=0.5)

    def recursive_unit(self, parent_word, child_hs, child_cells):
        def manyU_mul_manyH(manyU):
            sum = manyU[0].mul(child_hs[0]).sum(dim=1)
            for i in range(1, self.degree):
                sum += manyU[i].mul(child_hs[i]).sum(dim=1)
            return sum

        def forget_manycells(f_gates):
            sum = f_gates[0].mul(child_cells[0])
            for i in range(1, self.degree):
                sum += f_gates[i].mul(child_cells[i])
            return sum

        input = F.sigmoid(self.W_i.mul(parent_word).sum(dim=1) + manyU_mul_manyH(self.U_i) + self.b_i)
        output = F.sigmoid(self.W_o.mul(parent_word).sum(dim=1) + manyU_mul_manyH(self.U_o) + self.b_o)
        utility = F.tanh(self.W_u.mul(parent_word).sum(dim=1) + manyU_mul_manyH(self.U_u) + self.b_u)
        forgets = [F.sigmoid(self.W_f.mul(parent_word).sum(dim=1) + manyU_mul_manyH(this_U_f) + self.b_f) for
                   this_U_f in self.U_f]
        parent_cell = input.mul(utility) + forget_manycells(forgets)
        parent_h = output.mul(F.tanh(parent_cell))

        return parent_h, parent_cell