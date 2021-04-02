"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import time
import itertools
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pysnooper

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    @pysnooper.snoop('./res/debug.log')
    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c'], 'max_h':edges.src['max_h']}

    @pysnooper.snoop('./res/debug.log')
    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)
        max_h = nodes.mailbox['max_h'].max(dim=1)[0]
        return {'iou': self.U_iou(h_cat), 'c': c, 'max_h': max_h}

    @pysnooper.snoop('./res/debug.log')
    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        max_h = th.max(h, nodes.data['max_h'])
        return {'h' : h, 'c' : c, 'max_h':max_h}

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}

class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 device,
                 cell_type='nary',
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.device = device
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        cell = TreeLSTMCell if cell_type == 'nary' else ChildSumTreeLSTMCell
        self.cell = cell(x_size, h_size)

    def forward(self, batch):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g = batch.graph
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        embeds = self.embedding(batch.wordid * batch.mask)
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) * batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = th.zeros((g.number_of_nodes(), self.x_size)).to(self.device)
        g.ndata['c'] = th.zeros((g.number_of_nodes(), self.x_size)).to(self.device)
        g.ndata['max_h'] = th.zeros((g.number_of_nodes(), self.x_size)).to(self.device)
        # propagate
        dgl.prop_nodes_topo(g)
        # compute logits
        h = self.dropout(g.ndata.pop('h') + g.ndata.pop('max_h'))
        logits = self.linear(h)
        return logits

import torch as th
import pysnooper
import Transformer_Utils
# cell = TreeLSTMCell(256, 256)
# print("num_nodes:", graph.number_of_nodes())
# graph.ndata['h'] = th.ones(graph.number_of_nodes(), 256)
# graph.ndata['c'] = th.ones(graph.number_of_nodes(), 256)
# graph.ndata['iou'] = th.ones(graph.number_of_nodes(), 256*3)
# graph.register_message_func(cell.message_func)
# graph.register_reduce_func(cell.reduce_func)
# graph.register_apply_node_func(cell.apply_node_func)
#
# traversal_order = dgl.topological_nodes_generator(graph)
# graph.prop_nodes(traversal_order)
# print("is mul_graph:", graph.is_multigraph)
# the following is a syntax sugar that does the same
# dgl.prop_nodes_topo(graph)

##############################################################################
# .. note::
#
#    Before we call :meth:`~dgl.DGLGraph.prop_nodes`, we must specify a
#    `message_func` and `reduce_func` in advance, here we use built-in
#    copy-from-source and sum function as our message function and reduce
#    function for demonstration.
#
# Putting it together
# -------------------
#
# Here is the complete code that specifies the ``Tree-LSTM`` class:
#


class TransformerCell(nn.Module):
    def __init__(self, h_model, head=8):
        super(TransformerCell, self).__init__()
        self.head = head
        self.h_model = h_model
        self.attnLayer1 = Transformer_Utils.MultiHeadedAttention(self.head, h_model)
        self.attnLayer2 = Transformer_Utils.MultiHeadedAttention(self.head, h_model)
        self.layerNorm1 = Transformer_Utils.LayerNorm(h_model)
        self.layerNorm2 = Transformer_Utils.LayerNorm(h_model)

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h']
        return {'c': h_cat}

    def apply_node_func(self, nodes):
        ctx = th.cat(
            (nodes.data['c'].view(nodes.data['c'].size(0), -1), nodes.data['h'], nodes.data['e'], nodes.data['s']), -1
        ).view(nodes.data['c'].size(0), -1, self.h_model)
        h = self.attnLayer1(nodes.data['h'].unsqueeze(1), ctx, ctx)  # h: [nodes, dmodel] -> [nodes, 1, dmodel]  ctx: [nodes, kpairs, d_model]  --> [nodes, 1, dmodel]
        return {'h' : self.layerNorm1(F.relu(h.squeeze(1)))}

    def updateGlobalVec(self, S,  H):
        assert H.dim() == 3 and S.dim() == 3 #H: [tree, nodes, vector], S=[ tree, 1, vector ]
        ctx = th.cat(
            (H, S), dim=1
        )  # [tree, nodes+1, vector]
        States = self.layerNorm2(
            F.relu(
                self.attnLayer2(S, ctx, ctx, self_mask = True)
            )
        )  # [tree, 1, dmodel] [tree, nodes+1, dmodel] ---> [tree, 1, dmodel]
        return States.squeeze(1)   #[tree, 1, dmodel] -> [tree, dmodel]


class GraphTransformer(nn.Module):
    def __init__(self,
                 num_vocabs,
                 dmodel,
                 num_classes,
                 dropout,
                 device,
                 T_step = 5,
                 pretrained_emb=None):
        super(GraphTransformer, self).__init__()
        self.dmodel = dmodel
        self.device = device
        self.T_step = T_step
        self.embedding = nn.Embedding(num_vocabs, dmodel)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dmodel, num_classes)
        self.cell = TransformerCell(dmodel)

    def forward(self, batch):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        #----------utils function---------------
        def InitS(tree):
            tree.ndata['s'] = tree.ndata['e'].mean(dim=0).repeat(tree.number_of_nodes(), 1)
            return tree

        def updateS(tree, state):
            assert state.dim() == 1
            tree.ndata['s'] = state.repeat(tree.number_of_nodes(), 1)
            return tree

        def extractS(batchTree):
            # [dmodel] --> [[dmodel]] --> [tree, dmodel] --> [tree, 1, dmodel]
            s_list = [tree.ndata.pop('s')[0].unsqueeze(0) for tree in dgl.unbatch(batchTree)]
            return th.cat(s_list, dim=0).unsqueeze(1)

        def extractH(batchTree):
            # [nodes, dmodel] --> [nodes, dmodel]--> [max_nodes, dmodel]--> [tree*_max_nodes, dmodel] --> [tree, max_nodes, dmodel]
            h_list = [tree.ndata.pop('h') for tree in dgl.unbatch(batchTree)]
            max_nodes = max([h.size(0) for h in h_list])
            h_list = [th.cat([h, th.zeros([max_nodes-h.size(0), h.size(1)]).to(self.device)], dim=0).unsqueeze(0) for h in h_list]
            return th.cat(h_list, dim=0)
        #-----------------------------------------

        g = batch.graph
        # feed embedding
        embeds = self.embedding(batch.wordid * batch.mask)
        g.ndata['c'] = th.zeros((g.number_of_nodes(), 2, self.dmodel)).to(self.device)
        g.ndata['e'] = embeds*batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = embeds*batch.mask.float().unsqueeze(-1)
        g = dgl.batch([InitS(gg) for gg in dgl.unbatch(g)])
        # propagate
        for i in range(self.T_step):
            g.register_message_func(self.cell.message_func)
            g.register_reduce_func(self.cell.reduce_func)
            g.register_apply_node_func(self.cell.apply_node_func)
            dgl.prop_nodes_topo(g)
            States = self.cell.updateGlobalVec(extractS(g), extractH(g) )
            g = dgl.batch([updateS(tree, state) for (tree, state) in zip(dgl.unbatch(g), States)])
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        logits = self.linear(h)
        return logits