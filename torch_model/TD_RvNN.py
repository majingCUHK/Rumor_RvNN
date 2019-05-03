__doc__ = """Tree GRU aka Recursive Neural Networks."""

import numpy as np
import theano
from theano import tensor as T
#from collections import OrderedDict
from theano.compat.python2x import OrderedDict
from theano.tensor.signal.pool import pool_2d


theano.config.floatX = 'float32'


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        #self.index = index
        self.idx = idx
        self.word = []
        self.index = []
        #self.height = 1
        #self.size = 1
        #self.num_leaves = 1
        self.parent = None
        #self.label = None
        
################################# generate tree structure ##############################
#def gen_nn_inputs(root_node, ini_word, ini_index):
def gen_nn_inputs(root_node, ini_word):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    #_clear_indices(root_node)
    #x, leaf_labels = _get_leaf_vals(root_node)
    #X_word, X_index = _get_leaf_vals(root_node)
    tree = [[0, root_node.idx]] 
    #X_word, X_index = [ini_word], [ini_index]
    X_word, X_index = [root_node.word], [root_node.index]
    #print X_index
    #print X_word
    #exit(0)
    internal_tree, internal_word, internal_index  = _get_tree_path(root_node)
    #print internal_tree
    #assert all(v is not None for v in x)
    #if not only_leaves_have_vals:
    #    assert all(v is not None for v in internal_x)
    tree.extend(internal_tree)    
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    X_word.append(ini_word)
    #if max_degree is not None:
    #    assert all(len(t) == max_degree + 1 for t in tree)
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
    #print X_word    
    #print type(X_word)    
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'))
    #return (np.array(X_word),
    #        np.array(X_index),
    #        np.array(tree))        

def _get_tree_path(root_node):
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
    #print 'layer:', layers
    tree = []
    word = []
    index = []
    for layer in layers:
        for node in layer:
            if not node.children:
               continue 
            #child_idxs = [child.idx for child in ]  ## idx of child node
            for child in node.children:
                tree.append([node.idx, child.idx])
                word.append(child.word if child.word is not None else -1)
                index.append(child.index if child.index is not None else -1)
            '''if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_word.append(node.word if node.word is not None else -1)
            internal_index.append(node.index if node.index is not None else -1)
            idx += 1'''

    return tree, word, index

################################ tree rnn class ######################################
class RvNN(object):
    """Data is represented in a tree structure.

    Every leaf and internal node has a data (provided by the input)
    and a memory or hidden state.  The hidden state is computed based
    on its own data and the hidden states of its children.  The
    hidden state of leaves is given by a custom init function.

    The entire tree's embedding is represented by the final
    state computed at the root.

    """
    def __init__(self, word_dim, hidden_dim=5, Nclass=4,
                degree=2, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):                 
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree
        #self.learning_rate = learning_rate
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        self.params = []

        #self.x = T.ivector(name='x')  # word indices
        #self.x_word = T.matrix(dtype=theano.config.floatX)  # word frequendtype=theano.config.floatX
        self.x_word = T.matrix(name='x_word')  # word frequent
        self.x_index = T.imatrix(name='x_index')  # word indices
        self.tree = T.imatrix(name='tree')  # shape [None, self.degree]
        self.y = T.ivector(name='y')  # output shape [self.output_dim]
        self.num_parent = T.iscalar(name='num_parent')
        self.num_nodes = self.x_word.shape[0]  # total number of nodes (leaves + internal) in tree
        self.num_child = self.num_nodes - self.num_parent-1
        #emb_x = self.embeddings[self.x]
        #emb_x = emb_x * T.neq(self.x, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings

        self.tree_states = self.compute_tree(self.x_word, self.x_index, self.num_parent, self.tree)
        #self.final_state = self.tree_states.mean(axis=0)#self.tree_states[-1]
        #self.final_state = pool_2d(input=self.tree_states, ds=(self.num_child,1), ignore_border=True,mode='max')
        self.final_state = self.tree_states.max(axis=0)
        self.output_fn = self.create_output_fn()
        self.pred_y = self.output_fn(self.final_state)
        self.loss = self.loss_fn(self.y, self.pred_y)

        self.learning_rate = T.scalar('learning_rate')
        #updates = self.gradient_descent(self.loss, self.learning_rate)
        train_inputs = [self.x_word, self.x_index, self.num_parent, self.tree, self.y, self.learning_rate]
        updates = self.gradient_descent(self.loss)

        #train_inputs = [self.x_word, self.x_index, self.tree, self.y]
        self._train = theano.function(train_inputs,
                                      [self.loss, self.pred_y],
                                      updates=updates)

        self._evaluate = theano.function([self.x_word, self.x_index, self.num_parent, self.tree], self.final_state)
        self._evaluate2 = theano.function([self.x_word, self.x_index, self.num_parent, self.tree], self.tree_states)
        #self._state = theano.function([self.x_word, self.x_index, self.num_child, self.tree], self.tree_states)

        self._predict = theano.function([self.x_word, self.x_index, self.num_parent, self.tree], self.pred_y)
        
        self.tree_states_test = self.compute_tree_test(self.x_word, self.x_index, self.tree)
        self._evaluate3 = theano.function([self.x_word, self.x_index, self.tree], self.tree_states_test)

    '''def _check_input(self, x, tree):
        assert np.array_equal(tree[:, -1], np.arange(len(x) - len(tree), len(x)))
        if not self.irregular_tree:
            assert np.all((tree[:, 0] + 1 >= np.arange(len(tree))) |
                          (tree[:, 0] == -1))
            assert np.all((tree[:, 1] + 1 >= np.arange(len(tree))) |
                          (tree[:, 1] == -1))

    def train_step_inner(self, x, tree, y):
        self._check_input(x, tree)
        return self._train(x, tree[:, :-1], y)

    def train_step(self, root_node, y):
        x_word, x_index, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        return self._train(x_word, x_index, tree[:, :-1], y)
        #return self.train_step_inner(x, tree, y)'''
    
    def train_step_up(self, x_word, x_index, num_parent, tree, y, lr):
        #x_word, x_index, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        return self._train(x_word, x_index, num_parent, tree, y, lr)
        
    def evaluate(self,  x_word, x_index, num_parent, tree):
        #x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        #self._check_input(x, tree)
        return self._evaluate(x_word, x_index, num_parent, tree)

    def predict_up(self, x_word, x_index, num_parent, tree):
        #x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        #self._check_input(x, tree)
        return self._predict(x_word, x_index, num_parent, tree)

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)

    def create_output_fn(self):
        self.W_out = theano.shared(self.init_matrix([self.Nclass, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.Nclass]))
        self.params.extend([self.W_out, self.b_out])

        def fn(final_state):
            return T.nnet.softmax( self.W_out.dot(final_state)+ self.b_out )
        return fn

    '''def create_output_fn_multi(self):
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.W_out, self.b_out])

        def fn(tree_states):
            return T.nnet.softmax(
                T.dot(tree_states, self.W_out.T) +
                self.b_out.dimshuffle('x', 0))
        return fn'''

    def create_recursive_unit(self):
        self.E = theano.shared(self.init_matrix([self.hidden_dim, self.word_dim]))
        self.W_z = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_z = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_z = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_r = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_r = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_r = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_h = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_h = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([self.E, self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r, self.W_h, self.U_h, self.b_h])
        def unit(word, index, parent_h):
            #h_tilde = T.sum(child_h, axis=0)
            child_xe = self.E[:,index].dot(word)
            z = T.nnet.hard_sigmoid(self.W_z.dot(child_xe)+self.U_z.dot(parent_h)+self.b_z)
            r = T.nnet.hard_sigmoid(self.W_r.dot(child_xe)+self.U_r.dot(parent_h)+self.b_r)
            c = T.tanh(self.W_h.dot(child_xe)+self.U_h.dot(parent_h * r)+self.b_h)
            h = z*parent_h + (1-z)*c
            return h
        return unit

    '''def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))
        def unit(leaf_word, leaf_index):
            return self.recursive_unit( leaf_word, leaf_index, dummy, dummy.sum(axis=1))
        return unit'''

    def compute_tree(self, x_word, x_index, num_parent, tree):
        self.recursive_unit = self.create_recursive_unit()
        #num_nodes = self.num_nodes+1
        def ini_unit(x):
            return theano.shared(self.init_vector([self.hidden_dim]))
        #init_node_h = 0 * theano.shared(self.init_vector([self.num_nodes, self.hidden_dim]))        
        init_node_h, _ = theano.scan(
            fn=ini_unit,
            sequences=[ x_word ])
            #n_steps=num_nodes)
        #dummy = 0 * theano.shared(self.init_vector([self.hidden_dim]))        
        #init_node_h = T.concatenate([dummy, all_node_h], axis=0)
        
        '''self.recursive_unit = self.create_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        num_parents = tree.shape[0]  # num internal nodes
        num_leaves = self.num_nodes - num_parents

        # compute leaf hidden states
        leaf_h, _ = theano.map(
            fn=self.leaf_unit,
            sequences=[ x_word[:num_leaves], x_index[:num_leaves] ])
        if self.irregular_tree:
            init_node_h = T.concatenate([leaf_h, leaf_h, leaf_h], axis=0)
        else:
            init_node_h = leaf_h'''

        # use recurrence to compute internal node hidden states
        def _recurrence(x_word, x_index, node_info, node_h, last_h):
            parent_h = node_h[node_info[0]]
            child_h = self.recursive_unit(x_word, x_index, parent_h)
            #node_h[node_info[1]] = child_h
            node_h = T.concatenate([node_h[:node_info[1]],
                                    child_h.reshape([1, self.hidden_dim]),
                                    node_h[node_info[1]+1:] ])
            '''#child_exists = node_info > -1
            #offset = 2*num_leaves * int(self.irregular_tree) - child_exists * t ### offset???
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x') ### transpose??
            parent_h = self.recursive_unit(x_word, x_index, child_h, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            return node_h[1:], parent_h'''
            return node_h, child_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, child_hs), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[x_word[:-1], x_index, tree])

        '''dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[x_word[num_leaves:], x_index[num_leaves:], tree, T.arange(num_parents)],
            n_steps=num_parents)

        return T.concatenate([leaf_h, parent_h], axis=0)'''
        return child_hs[num_parent-1:]

    def compute_tree_test(self, x_word, x_index, tree):
        self.recursive_unit = self.create_recursive_unit()
        def ini_unit(x):
            return theano.shared(self.init_vector([self.hidden_dim]))
        init_node_h, _ = theano.scan(
            fn=ini_unit,
            sequences=[ x_word ])

        def _recurrence(x_word, x_index, node_info, node_h, last_h):
            parent_h = node_h[node_info[0]]
            child_h = self.recursive_unit(x_word, x_index, parent_h)
            #node_h[node_info[1]] = child_h
            node_h = T.concatenate([node_h[:node_info[1]],
                                    child_h.reshape([1, self.hidden_dim]),
                                    node_h[node_info[1]+1:] ])
            return node_h, child_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, child_hs), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[x_word[:-1], x_index, tree])
        return child_hs
        
    def loss_fn(self, y, pred_y):
        return T.sum(T.sqr(y - pred_y))

    '''def loss_fn_multi(self, y, pred_y, y_exists):
        return T.sum(T.sum(T.sqr(y - pred_y), axis=1) * y_exists, axis=0)'''

    def gradient_descent(self, loss):
        """Momentum GD with gradient clipping."""
        grad = T.grad(loss, self.params)
        self.momentum_velocity_ = [0.] * len(grad)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grad)))
        updates = OrderedDict()
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den = T.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(self.params, grad)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (5.0 / scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - self.learning_rate * grad
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates
        