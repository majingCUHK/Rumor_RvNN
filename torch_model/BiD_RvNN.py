# -*- coding: utf-8 -*-

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RvNN(nn.Module):
    def __init__(self, word_dim, hidden_dim=5, Nclass=4,
                 degree=2, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):
        super(RvNN, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree  # 这里比较奇怪的是，在创建模型的时候是没有对degree进行赋值的
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        self.E_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.word_dim]), requires_grad=True)
        self.W_z_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_z_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_z_td = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_r_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_r_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_r_td = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        self.W_h_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.U_h_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]), requires_grad=True)
        self.b_h_td = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)

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

        self.W_out1 = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, 2*self.hidden_dim]), requires_grad=True)
        self.b_out1 = nn.parameter.Parameter(self.init_vector([self.hidden_dim]), requires_grad=True)
        # self.W_out2 = nn.parameter.Parameter(self.init_matrix([2*self.hidden_dim, 2*self.hidden_dim]), requires_grad=True)
        # self.b_out2 = nn.parameter.Parameter(self.init_vector([2*self.hidden_dim]), requires_grad=True)
        # self.W_out3 = nn.parameter.Parameter(self.init_matrix([2*self.hidden_dim, 2*self.hidden_dim]), requires_grad=True)
        # self.b_out3 = nn.parameter.Parameter(self.init_vector([2*self.hidden_dim]), requires_grad=True)
        self.W_out4 = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad=True)
        self.b_out4 = nn.parameter.Parameter(self.init_vector([self.Nclass]), requires_grad=True)

    def forward(self, td_x_word, td_x_index, td_tree):
        td_final_state = self.td_compute_tree_states(td_x_word, td_x_index, td_tree)
        bu_final_state = self.bu_compute_tree_states(td_x_word, td_x_index, td_tree)
        final_state = torch.cat((td_final_state, bu_final_state), dim=0)
        final_state1 = F.relu(self.W_out1.mul(final_state).sum(dim=1) +self.b_out1)
        # final_state2 = self.W_out2.mul(final_state1).sum(dim=1) + self.b_out2
        # final_state3 = self.W_out3.mul(final_state2).sum(dim=1) + self.b_out3
        pred = F.softmax(self.W_out4.mul(final_state1).sum(dim=1) +self.b_out4)
        return pred

    def td_recursive_unit(self, child_word, child_index, parent_h):
        child_xe = self.E_td[:, child_index].mul(torch.tensor(child_word)).sum(dim=1)
        z_td = F.sigmoid(self.W_z_td.mul(child_xe).sum(dim=1) + self.U_z_td.mul(parent_h).sum(dim=1) + self.b_z_td)
        r_td = F.sigmoid(self.W_r_td.mul(child_xe).sum(dim=1) + self.U_r_td.mul(parent_h).sum(dim=1) + self.b_r_td)
        c = F.tanh(self.W_h_td.mul(child_xe).sum(dim=1) + self.U_h_td.mul(parent_h * r_td).sum(dim=1) + self.b_h_td)
        h_td = z_td * parent_h + (1 - z_td) * c
        return h_td

    def td_compute_tree_states(self, x_word, x_index, tree, leaf_idxs):

        def _recurrence(x_word, x_index, tree, node_h):
            parent_h = node_h[tree[0]]
            child_h = self.td_recursive_unit(x_word, x_index, parent_h)
            node_h = torch.cat((node_h, child_h.view(1, -1)))
            return node_h

        node_h = torch.zeros([1, self.hidden_dim])

        for words, indexs, thislayer in zip(x_word, x_index, tree):
            node_h = _recurrence(words, indexs, thislayer, node_h)

        return node_h[leaf_idxs].max(dim=0)[0]

    def bu_recursive_unit(self, parent_word, parent_index, child_h):
        h_tilde = child_h.sum(dim=0)
        parent_xe = self.E_bu[:, parent_index].mul(torch.tensor(parent_word)).sum(dim=1)
        z_bu = F.sigmoid(self.W_z_bu.mul(parent_xe).sum(dim=1) + self.U_z_bu.mul(h_tilde).sum(dim=1) + self.b_z_bu)
        r_bu = F.sigmoid(self.W_r_bu.mul(parent_xe).sum(dim=1) + self.U_r_bu.mul(h_tilde).sum(dim=1) + self.b_r_bu)
        c = F.tanh(self.W_h_bu.mul(parent_xe).sum(dim=1) + self.U_h_bu.mul(h_tilde * r_bu).sum(dim=1) + self.b_h_bu)
        h_bu = z_bu * h_tilde + (1 - z_bu) * c
        return h_bu

    def bu_compute_tree_states(self, x_word, x_index, tree):
        num_parents = tree.shape[0]
        num_nodes = x_word.shape[0]
        num_leaves = num_nodes -num_parents
        leaf_h = list(map(
                            lambda x_idxs: self.bu_recursive_unit(x_idxs[0], x_idxs[1], torch.zeros([self.degree, self.hidden_dim])).tolist(),
                                zip(x_word[:num_leaves], x_index[:num_leaves])
                        )
                    )

        init_node_h = torch.tensor(leaf_h)

        def _recurrence(x_word, x_index, tree, idx, node_h):
            child_exists = (tree[:-1] > -1).nonzero()
            child_h = node_h[ tree[child_exists] ]
            parent_h = self.bu_recursive_unit(x_word, x_index, child_h)
            node_h = torch.cat((node_h, parent_h.view(1, -1)), 0)
            return node_h, parent_h

        node_h = init_node_h
        root_state = []
        for idx, (words, indexs, thislayer) in enumerate(zip(x_word[num_leaves:], x_index[num_leaves:], tree)):
            node_h, parent_h = _recurrence(words, indexs, thislayer, idx, node_h)
            if idx == num_parents-1:
                root_state = parent_h
        return root_state

    def init_vector(self, shape):
        return torch.zeros(shape)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, td_x_word, td_x_index, td_tree, td_leaf_idxs, bu_x_word, bu_x_index, bu_tree):
        td_final_state = self.td_compute_tree_states(td_x_word, td_x_index, td_tree, td_leaf_idxs)
        bu_final_state = self.bu_compute_tree_states(bu_x_word, bu_x_index, bu_tree)
        final_state = torch.cat((td_final_state, bu_final_state), dim=0)
        final_state1 = self.W_out1.mul(final_state).sum(dim=1) +self.b_out1
        # final_state2 = self.W_out2.mul(final_state1).sum(dim=1) + self.b_out2
        # final_state3 = self.W_out3.mul(final_state2).sum(dim=1) + self.b_out3
        return F.softmax(self.W_out4.mul(final_state1).sum(dim=1) +self.b_out4)