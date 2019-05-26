#-*- encoding: utf8

__doc__ = """Tree GRU aka Recursive Neural Networks."""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.b_z_bu = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_r_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_r_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_r_bu = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_h_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_h_bu = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_h_bu = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_out_bu = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad=True)
        self.b_out_bu = nn.parameter.Parameter(self.init_vector([self.Nclass]), requires_grad=True)

        self.WQ = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.WK = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.WV = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))

    def forward(self, x_word, x_index, tree, y):
        final_state = self.compute_tree_states(x_word, x_index, tree)
        pred, loss = self.predAndLoss(final_state, y)
        return pred, loss

    def recursive_unit(self, parent_word, parent_index, child_h):
        #attention
        parent_xe = self.E_bu[:, parent_index].mul(torch.tensor(parent_word)).sum(dim=1)
        query = F.sigmoid(parent_xe.mul(self.WQ.t()).sum(dim=1))
        key = child_h.mm(self.WK)
        val = child_h.mm(self.WV)
        attention = F.softmax( (query/np.sqrt(self.hidden_dim*1.0)).mul(key).sum(dim=1) )
        h_tilde = attention.mul(val.t()).sum(dim=1)
        #gru
        z_bu = F.sigmoid(self.W_z_bu.mul(parent_xe).sum(dim=1) + self.U_z_bu.mul(h_tilde).sum(dim=1) + self.b_z_bu)
        r_bu = F.sigmoid(self.W_r_bu.mul(parent_xe).sum(dim=1) + self.U_r_bu.mul(h_tilde).sum(dim=1) + self.b_r_bu)
        c = F.tanh(self.W_h_bu.mul(parent_xe).sum(dim=1) + self.U_h_bu.mul(h_tilde * r_bu).sum(dim=1) + self.b_h_bu)
        h_bu = z_bu * h_tilde + (1 - z_bu) * c
        return h_bu

    def compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes -num_parents
        leaf_h = list(map(
                            lambda x_idxs: self.recursive_unit(x_idxs[0], x_idxs[1], torch.zeros([self.degree, self.hidden_dim])).tolist(),
                                zip(x_word[:num_leaves], x_index[:num_leaves])
                        )
                    )

        init_node_h = torch.tensor(leaf_h)

        def _recurrence(x_word, x_index, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            child_h = node_h[ tree[child_exists] ]
            parent_h = self.recursive_unit(x_word, x_index, child_h)
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

    def predict_up(self, x_word, x_index, x_tree):
        final_state = self.compute_tree_states(x_word, x_index, x_tree)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)

class MultiAttentionGRU(nn.Module):
    def __init__(self, word_dim, hidden_dim=5, Nclass=4,
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
    def forward(self, x_word, x_index, tree, y):
        final_state = self.compute_tree_states(x_word, x_index, tree)
        pred, loss = self.predAndLoss(final_state, y)
        return pred, loss

    def recursive_unit(self, parent_word, parent_index, child_h):
        #attention
        parent_xe = self.E_bu[:, parent_index].mul(torch.tensor(parent_word)).sum(dim=1)

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

        h_tilde = F.sigmoid(self.WO.mul(h_tilde).sum(dim=1))

        #gru
        z_bu = F.sigmoid(self.W_z_bu.mul(parent_xe).sum(dim=1) + self.U_z_bu.mul(h_tilde).sum(dim=1) + self.b_z_bu)
        r_bu = F.sigmoid(self.W_r_bu.mul(parent_xe).sum(dim=1) + self.U_r_bu.mul(h_tilde).sum(dim=1) + self.b_r_bu)
        c = F.tanh(self.W_h_bu.mul(parent_xe).sum(dim=1) + self.U_h_bu.mul(h_tilde * r_bu).sum(dim=1) + self.b_h_bu)
        h_bu = z_bu * h_tilde + (1 - z_bu) * c
        return h_bu

    def compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes -num_parents
        leaf_h = list(map(
                            lambda x_idxs: self.recursive_unit(x_idxs[0], x_idxs[1], torch.zeros([self.degree, self.hidden_dim])).tolist(),
                                zip(x_word[:num_leaves], x_index[:num_leaves])
                        )
                    )

        init_node_h = torch.tensor(leaf_h)

        def _recurrence(x_word, x_index, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            child_h = node_h[ tree[child_exists] ]
            parent_h = self.recursive_unit(x_word, x_index, child_h)
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

    def predict_up(self, x_word, x_index, x_tree):
        final_state = self.compute_tree_states(x_word, x_index, x_tree)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)

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
        return torch.zeros(shape).cuda()

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32')).cuda()

    def predict_up(self, x_word, x_index, x_tree):
        final_state = self.compute_tree_states(x_word, x_index, x_tree)
        return F.softmax(self.W_out_bu.mul(final_state).sum(dim=1) +self.b_out_bu)