import pandas as pd
from math import log2

# used for draw the graph.
from queue import Queue
import networkx as nx
import matplotlib.pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout

plt.rcParams['figure.figsize'] = (25, 25)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def calculate_gini_value(data, pred_col_name='人'):
    result = 1
    for count in data[pred_col_name].value_counts():
        result -= (count / len(data)) ** 2
    return result


def calculate_entropy_value(data, pred_col_name='人'):
    result = 0
    #     print(data.shape)
    for count in data[pred_col_name].value_counts():
        result -= (count / len(data)) * log2(count / len(data))
    return result


class TreeNode():
    def __init__(self, d_set=None, truebranch=None, falsebranch=None,
                 split_attr=None):
        self.d_set = d_set
        self.truebranch = truebranch
        self.falsebranch = falsebranch
        self.split_attr = split_attr


class DecisionTree():
    def __init__(self, d_set=None, pred_col_name=None, evaluate_function=None):
        self.dataset = d_set
        self.column = pred_col_name
        self.eval_func = evaluate_function
        self.root = None

    def fit(self):

        def generate_decision_tree(dataset, evaluateFunction=calculate_entropy_value):
            if dataset.shape[0] == 1:
                return TreeNode(d_set=dataset)
            # calculate the cross entropy of current dataset.
            dataset_value = evaluateFunction(dataset)

            max_gain = 0
            best_subset = None
            best_split_attr = None

            for split_attr in dataset.columns[1:]:
                truthbranch = dataset[dataset[split_attr] == '是']
                truthbranch.index = range(len(truthbranch))
                falsebranch = dataset[dataset[split_attr] == '否']
                falsebranch.index = range(len(falsebranch))

                truth_partion = len(truthbranch) / len(dataset)
                false_partion = 1 - truth_partion
                truth_value = evaluateFunction(truthbranch)
                false_value = evaluateFunction(falsebranch)

                gain = dataset_value - (truth_partion * truth_value) - (false_value * false_partion)
                if gain > max_gain:
                    max_gain = gain
                    best_subset = (truthbranch, falsebranch)
                    best_split_attr = split_attr

            dcY = {'impurity': '%.3f' % dataset_value, 'samples': '%d' % len(dataset)}
            print(f"split_attribute : {best_split_attr}, max_gain : {max_gain}")
            print(dcY)
            if max_gain > 0:
                truebranch = generate_decision_tree(best_subset[0], evaluateFunction)
                falsebranch = generate_decision_tree(best_subset[1], evaluateFunction)
                return TreeNode(d_set=dataset, truebranch=truebranch,
                                falsebranch=falsebranch,
                                split_attr=best_split_attr)
            else:
                return TreeNode(d_set=dataset)


        self.root = generate_decision_tree(self.dataset,
                                           evaluateFunction=self.eval_func)

    def draw(self):

        G = nx.DiGraph()
        node_q = Queue()
        G.add_node("Node 1\n" + self.root.split_attr)
        node_q.put(("Node 1\n" + self.root.split_attr, self.root))
        counter = 1
        while not node_q.empty():
            current_name, current_node = node_q.get()
            if current_node.truebranch.split_attr == None:
                G.add_node(current_node.truebranch.d_set['人'][0], color='red', time='20pt')
                G.add_edge(current_name,
                           current_node.truebranch.d_set['人'][0])
            else:
                counter += 1
                G.add_node('Node ' + str(counter) + '\n' + current_node.truebranch.split_attr)
                node_q.put(
                    ('Node ' + str(counter) + '\n' + current_node.truebranch.split_attr, current_node.truebranch))
                G.add_edge(current_name, 'Node ' + str(counter) + '\n' + current_node.truebranch.split_attr)
            if current_node.falsebranch.split_attr == None:
                G.add_node(current_node.falsebranch.d_set['人'][0], color='red', time='20pt')
                G.add_edge(current_name,
                           current_node.falsebranch.d_set['人'][0])
            else:
                counter += 1
                G.add_node('Node ' + str(counter) + '\n' + current_node.falsebranch.split_attr, color='orange',
                           time='20pt')
                node_q.put(
                    ('Node ' + str(counter) + '\n' + current_node.falsebranch.split_attr, current_node.falsebranch))
                G.add_edge(current_name, 'Node ' + str(counter) + '\n' + current_node.falsebranch.split_attr)
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=True, arrows=True, node_size=10000, font_size=40)
        plt.show()


if __name__ == '__main__':
    data_input = pd.read_csv('./data/data.csv')
    model = DecisionTree(d_set=data_input, pred_col_name='人',
                         evaluate_function=calculate_entropy_value)
    model.fit()
    model.draw()
