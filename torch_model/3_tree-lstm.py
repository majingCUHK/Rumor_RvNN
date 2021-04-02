"""
.. _model-tree-lstm:

Tree LSTM DGL Tutorial
=========================

**Author**: Zihao Ye, Qipeng Guo, `Minjie Wang
<https://jermainewang.github.io/>`_, `Jake Zhao
<https://cs.nyu.edu/~jakezhao/>`_, Zheng Zhang
"""
 
##############################################################################
#
# Tree-LSTM structure was first introduced by Kai et. al in an ACL 2015 
# paper: `Improved Semantic Representations From Tree-Structured Long
# Short-Term Memory Networks <https://arxiv.org/pdf/1503.00075.pdf>`__.
# The core idea is to introduce syntactic information for language tasks by 
# extending the chain-structured LSTM to a tree-structured LSTM. The Dependency 
# Tree/Constituency Tree techniques were leveraged to obtain a ''latent tree''.
#
# One, if not all, difficulty of training Tree-LSTMs is batching --- a standard 
# technique in machine learning to accelerate optimization. However, since trees 
# generally have different shapes by nature, parallization becomes non trivial. 
# DGL offers an alternative: to pool all the trees into one single graph then 
# induce the message passing over them guided by the structure of each tree.
#
# The task and the dataset
# ------------------------
# In this tutorial, we will use Tree-LSTMs for sentiment analysis.
# We have wrapped the
# `Stanford Sentiment Treebank <https://nlp.stanford.edu/sentiment/>`__ in
# ``dgl.data``. The dataset provides a fine-grained tree level sentiment
# annotation: 5 classes(very negative, negative, neutral, positive, and
# very positive) that indicates the sentiment in current subtree. Non-leaf
# nodes in constituency tree does not contain words, we use a special
# ``PAD_WORD`` token to denote them, during the training/inferencing,
# their embeddings would be masked to all-zero.
#
# .. figure:: https://i.loli.net/2018/11/08/5be3d4bfe031b.png
#    :alt: 
#
# The figure displays one sample of the SST dataset, which is a
# constituency parse tree with their nodes labeled with sentiment. To
# speed up things, let's build a tiny set with 5 sentences and take a look
# at the first one:
#

import dgl
import pysnooper
from dgl.data.tree import SST
from dgl.data import SSTBatch
import sys

# Each sample in the dataset is a constituency tree. The leaf nodes
# represent words. The word is a int value stored in the "x" field.
# The non-leaf nodes has a special word PAD_WORD. The sentiment
# label is stored in the "y" feature field.
trainset = SST()  # the "tiny" set has only 5 trees
tiny_sst = trainset.trees
num_vocabs = trainset.num_vocabs
num_classes = trainset.num_classes

vocab = trainset.vocab # vocabulary dict: key -> id
inv_vocab = {v: k for k, v in vocab.items()} # inverted vocabulary dict: id -> word

a_tree = tiny_sst[0]
for token in a_tree.ndata['x'].tolist():
    if token != trainset.PAD_WORD:
        print(inv_vocab[token], end=" ")

##############################################################################
# Step 1: batching
# ----------------
#
# The first step is to throw all the trees into one graph, using
# the :func:`~dgl.batched_graph.batch` API.
#

import networkx as nx
import matplotlib.pyplot as plt

graph = dgl.batch(tiny_sst)
def plot_tree(g):
    # this plot requires pygraphviz package
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=False, node_size=10,
            node_color=[[.5, .5, .5]], arrowsize=4)
    plt.show()
# plot_tree(graph.to_networkx())
##############################################################################
# You can read more about the definition of :func:`~dgl.batched_graph.batch`
# (by clicking the API), or can skip ahead to the next step:
# 
# .. note::
#
#    **Definition**: a :class:`~dgl.batched_graph.BatchedDGLGraph` is a
#    :class:`~dgl.DGLGraph` that unions a list of :class:`~dgl.DGLGraph`\ s. 
#    
#    - The union includes all the nodes,
#      edges, and their features. The order of nodes, edges and features are
#      preserved. 
#     
#        - Given that we have :math:`V_i` nodes for graph
#          :math:`\mathcal{G}_i`, the node ID :math:`j` in graph
#          :math:`\mathcal{G}_i` correspond to node ID
#          :math:`j + \sum_{k=1}^{i-1} V_k` in the batched graph. 
#    
#        - Therefore, performing feature transformation and message passing on
#          ``BatchedDGLGraph`` is equivalent to doing those
#          on all ``DGLGraph`` constituents in parallel. 
#
#    - Duplicate references to the same graph are
#      treated as deep copies; the nodes, edges, and features are duplicated,
#      and mutation on one reference does not affect the other. 
#    - Currently, ``BatchedDGLGraph`` is immutable in
#      graph structure (i.e. one can't add
#      nodes and edges to it). We need to support mutable batched graphs in
#      (far) future. 
#    - The ``BatchedDGLGraph`` keeps track of the meta
#      information of the constituents so it can be
#      :func:`~dgl.batched_graph.unbatch`\ ed to list of ``DGLGraph``\ s.
#
# For more details about the :class:`~dgl.batched_graph.BatchedDGLGraph`
# module in DGL, you can click the class name.
#
# Step 2: Tree-LSTM Cell with message-passing APIs
# ------------------------------------------------
#
# The authors proposed two types of Tree LSTM: Child-Sum
# Tree-LSTMs, and :math:`N`-ary Tree-LSTMs. In this tutorial we focus 
# on applying *Binary* Tree-LSTM to binarized constituency trees(this 
# application is also known as *Constituency Tree-LSTM*). We use PyTorch 
# as our backend framework to set up the network.
#
# In `N`-ary Tree LSTM, each unit at node :math:`j` maintains a hidden
# representation :math:`h_j` and a memory cell :math:`c_j`. The unit
# :math:`j` takes the input vector :math:`x_j` and the hidden
# representations of the their child units: :math:`h_{jl}, 1\leq l\leq N` as
# input, then update its new hidden representation :math:`h_j` and memory
# cell :math:`c_j` by: 
#
# .. math::
#
#    i_j & = & \sigma\left(W^{(i)}x_j + \sum_{l=1}^{N}U^{(i)}_l h_{jl} + b^{(i)}\right),  & (1)\\
#    f_{jk} & = & \sigma\left(W^{(f)}x_j + \sum_{l=1}^{N}U_{kl}^{(f)} h_{jl} + b^{(f)} \right), &  (2)\\
#    o_j & = & \sigma\left(W^{(o)}x_j + \sum_{l=1}^{N}U_{l}^{(o)} h_{jl} + b^{(o)} \right), & (3)  \\
#    u_j & = & \textrm{tanh}\left(W^{(u)}x_j + \sum_{l=1}^{N} U_l^{(u)}h_{jl} + b^{(u)} \right), & (4)\\
#    c_j & = & i_j \odot u_j + \sum_{l=1}^{N} f_{jl} \odot c_{jl}, &(5) \\
#    h_j & = & o_j \cdot \textrm{tanh}(c_j), &(6)  \\
#
# It can be decomposed into three phases: ``message_func``,
# ``reduce_func`` and ``apply_node_func``.
#
# .. note::
#    ``apply_node_func`` is a new node UDF we have not introduced before. In
#    ``apply_node_func``, user specifies what to do with node features,
#    without considering edge features and messages. In Tree-LSTM case,
#    ``apply_node_func`` is a must, since there exists (leaf) nodes with
#    :math:`0` incoming edges, which would not be updated via
#    ``reduce_func``.
#

import torch as th
import torch.nn as nn

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        print("h_cat:", *nodes.mailbox['h'].size())
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        # equation (2)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        # print("f:", f.size())
        # second term of equation (5)
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}

##############################################################################
# Step 3: define traversal
# ------------------------
#
# After defining the message passing functions, we then need to induce the
# right order to trigger them. This is a significant departure from models
# such as GCN, where all nodes are pulling messages from upstream ones
# *simultaneously*.
#
# In the case of Tree-LSTM, messages start from leaves of the tree, and
# propagate/processed upwards until they reach the roots. A visualization
# is as follows:
#
# .. figure:: https://i.loli.net/2018/11/09/5be4b5d2df54d.gif
#    :alt:
#
# DGL defines a generator to perform the topological sort, each item is a
# tensor recording the nodes from bottom level to the roots. One can
# appreciate the degree of parallelism by inspecting the difference of the
# followings:
#

print('\nTraversing one tree:')
print(dgl.topological_nodes_generator(a_tree))
print('Traversing many trees at the same time:')
print(dgl.topological_nodes_generator(graph))

##############################################################################
# We then call :meth:`~dgl.DGLGraph.prop_nodes` to trigger the message passing:

import dgl.function as fn
import torch as th
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

    @pysnooper.snoop('./res/debug.log')
    def message_func(self, edges):
        return {'h': edges.src['h']}

    @pysnooper.snoop('./res/debug.log')
    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h']
        return {'c': h_cat}

    @pysnooper.snoop('./res/debug.log')
    def apply_node_func(self, graph):
        print("apply_node_func 1")
        ctx = th.cat(
            (graph.ndata['c'].view(graph.ndata['c'].size(0), -1), graph.ndata['h'], graph.ndata['e'], graph.ndata['s']), -1
        ).view(graph.ndata['c'].size(0), -1, self.h_model)
        print("apply_node_func 2")
        h = self.attnLayer1(graph.ndata['h'].unsqueeze(1), ctx, ctx)  # h: [nodes, dmodel] -> [nodes, 1, dmodel]  ctx: [nodes, kpairs, d_model]  --> [nodes, 1, dmodel]
        print("apply_node_func 3")
        return self.layerNorm1(F.relu(h.squeeze(1)))

    @pysnooper.snoop('./res/debug.log')
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
        @pysnooper.snoop('./res/debug.log')
        def InitS(tree):
            tree.ndata['s'] = tree.ndata['e'].mean(dim=0).repeat(tree.number_of_nodes(), 1)
            return tree

        @pysnooper.snoop('./res/debug.log')
        def updateS(tree, state):
            assert state.dim() == 1
            tree.ndata['s'] = state.repeat(tree.number_of_nodes(), 1)
            return tree

        @pysnooper.snoop('./res/debug.log')
        def extractS(batchTree):
            # [dmodel] --> [[dmodel]] --> [tree, dmodel] --> [tree, 1, dmodel]
            s_list = [tree.ndata.pop('s')[0].unsqueeze(0) for tree in dgl.unbatch(batchTree)]
            return th.cat(s_list, dim=0).unsqueeze(1)

        @pysnooper.snoop('./res/debug.log')
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
            # g.register_apply_node_func(self.cell.apply_node_func)
            print("prop_nodes_top %d" % i)
            dgl.prop_nodes_topo(g)
            print("prop_nodes_top %d completed"% i)
            h_new = self.cell.apply_node_func(g)
            print("h_new size:", h_new.size())
            g.ndata['h'] = h_new
            print("update node state completed at %d times!"%i)
            States = self.cell.updateGlobalVec(extractS(g), extractH(g) )
            print("update Global vec %d" % i)
            g = dgl.batch([updateS(tree, state) for (tree, state) in zip(dgl.unbatch(g), States)])
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        logits = self.linear(h)
        return logits



class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, batch, h, c):
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
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        logits = self.linear(h)
        return logits

##############################################################################
# Main Loop
# ---------
# Finally, we could write a training paradigm in PyTorch:
#

from torch.utils.data import DataLoader
import torch.nn.functional as F

device = th.device('cpu')
# hyper parameters
x_size = 256
h_size = 256
dropout = 0.2
lr = 0.05
weight_decay = 1e-4
epochs = 10

# create the model
model = GraphTransformer(trainset.num_vocabs,
                         x_size,
                         trainset.num_classes,
                         dropout,
                         device,
                         T_step = 5,
                         pretrained_emb = trainset.pretrained_emb).to(device)
print(model)

# create the optimizer
optimizer = th.optim.Adagrad(model.parameters(),
                          lr=lr,
                          weight_decay=weight_decay)

def batcher(dev):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(graph=batch_trees,
                        mask=batch_trees.ndata['mask'].to(dev),
                        wordid=batch_trees.ndata['x'].to(dev),
                        label=batch_trees.ndata['y'].to(dev))
    return batcher_dev

train_loader = DataLoader(dataset=tiny_sst,
                          batch_size= 5,
                          collate_fn=batcher(device),
                          shuffle=False,
                          num_workers=0)

# training loop
for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        g = batch.graph

        logits = model(batch)

        logp = F.log_softmax(logits, 1)

        loss = F.nll_loss(logp, batch.label, reduction='sum') 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = th.argmax(logits, 1)
        acc = float(th.sum(th.eq(batch.label, pred))) / len(batch.label)
        print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} |".format(
            epoch, step, loss.item(), acc))

##############################################################################
# To train the model on full dataset with different settings(CPU/GPU,
# etc.), please refer to our repo's
# `example <https://github.com/dmlc/dgl/tree/master/examples/pytorch/tree_lstm>`__.
# Besides, we also provide an implementation of the Child-Sum Tree LSTM.
