import networkx as nx
import matplotlib.pyplot as plt


class Tree(object):
    def __init__(self):
        super(Tree, self).__init__()
        self.tree = nx.DiGraph()
        self.tree.add_node(0) #root node
    def Parse_Sentence2Tree(self, sentence):
    #format: (3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) (2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 's)) (2 (3 new) (2 (2 ``) (2 Conan)))))))) (2 '')) (2 and)) (3 (2 that) (3 (2 he) (3 (2 's) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash)) (2 (2 even) (3 greater)))) (2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) (2 Schwarzenegger)) (2 ,)) (2 (2 Jean-Claud) (2 (2 Van) (2 Damme)))) (2 or)) (2 (2 Steven) (2 Segal))))))))))))) (2 .)))
        cur_idx = 0
        cur_node = 0
        node_num = 1
        assert sentence[cur_idx]=='('
        self.tree.add_node(0, label = int(sentence[cur_idx+1]))
        while True:
            start = sentence[cur_idx+1:].find('(')
            end = sentence[cur_idx+1:].find(')')
            if start ==end and end == -1:
                break
            if start == -1 or end < start:
                if sentence[end-1] != ')':
                    if sentence[cur_idx+2:end] != ' ':
                        self.tree.node[cur_node]['data'] = sentence[cur_idx+2:(cur_idx+1)+end] # cur_idx is the start of the current node
                if cur_node != 0: # if not the root node
                    cur_node = list(self.tree.predecessors(cur_node))[0]
                cur_idx += end+1
            elif start!= -1 and start < end:
                self.tree.add_node(node_num, label= int(sentence[(cur_idx+1)+(start+1)]))
                self.tree.add_edge(cur_node, node_num)
                cur_node = node_num
                node_num += 1
                cur_idx += start+1 # now, it indictes the index of this new '('
    def LeafNodes(self):
        leaf_nodes = []
        out_degrees = self.tree.out_degree()
        nodes = list(self.tree.nodes())
        for node in nodes:
            if out_degrees[node] == 0:
                leaf_nodes.append(node)
        return leaf_nodes


tree = Tree()
sentence = "(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) (2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 's)) (2 (3 new) (2 (2 ``) (2 Conan)))))))) (2 '')) (2 and)) (3 (2 that) (3 (2 he) (3 (2 's) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash)) (2 (2 even) (3 greater)))) (2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) (2 Schwarzenegger)) (2 ,)) (2 (2 Jean-Claud) (2 (2 Van) (2 Damme)))) (2 or)) (2 (2 Steven) (2 Segal))))))))))))) (2 .)))"
tree.Parse_Sentence2Tree(sentence)
print(tree.LeafNodes())
# tree.draw_tree()