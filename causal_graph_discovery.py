import copy
import copy
import csv
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.FCMBased import lingam
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


class CausalGraph:
    def __init__(self, causal_matrix=None, causal_dag=None, node_name_list=None, png_save_path='test_png'):
        '''
        :param causal_matrix: If A[i,j] !=0, the causal relationship: i -> j
        :param causal_graph: The causal graph object of the causal-learn library
        :param save_path: The save path of the png format of the obtained causal graph
        '''
        self.causal_matrix = causal_matrix
        self.causal_dag = causal_dag
        self.save_path = png_save_path
        self.node_name_list = node_name_list

    def fit_ica(self, data_set_path, le_data, current_seeds=3):
        cd_model = lingam.ICALiNGAM(random_state=current_seeds, max_iter=10000)
        cd_model.fit(le_data)
        weighted_causal_matrix = copy.deepcopy(cd_model.adjacency_matrix_)
        self.causal_matrix = weighted_causal_matrix.T
        self.get_node_names_list(data_set_path)
        self.create_dag_from_matrix(self.node_name_list)

    def get_node_names_list(self, data_set_path):
        with open(data_set_path, 'r', newline='', encoding='utf-8-sig') as file:
            csv_reader = csv.reader(file)
            node_names = next(csv_reader)
            node_names_list = list(map(str, node_names))

        self.node_names_list = node_names_list

    def create_dag_from_matrix(self, node_names_list):
        nodes = []
        causal_matrix = self.causal_matrix
        for name in node_names_list:
            node = GraphNode(name)
            nodes.append(node)

        dag = Dag(nodes)

        num_variables = causal_matrix.shape[0]

        for i in range(num_variables):
            for j in range(num_variables):
                if causal_matrix[i, j] != 0:
                    dag.add_directed_edge(nodes[i], nodes[j])

        self.causal_dag = dag

    def find_mb(self, x=-1):
        # 获取矩阵的维度
        causal_matrix = self.causal_matrix
        num_var = causal_matrix.shape[0]
        # 1. 获取X的所有父节点
        parents_of_X = [i for i in range(num_var) if causal_matrix[i, x] != 0]

        # 2. 获取X的所有子节点
        children_of_X = [i for i in range(num_var) if causal_matrix[x, i] != 0]

        # 3. 获取X的所有子节点的其他父节点
        spouses_of_X = []
        for child in children_of_X:
            spouses_of_X.extend([i for i in range(num_var) if causal_matrix[i, child] != 0 and i != x])

        # 合并所有的组件并去除重复的以及目标变量，即最后一个变量
        markov_blanket = set(parents_of_X + children_of_X + spouses_of_X)
        markov_blanket.discard(num_var - 1)

        return list(markov_blanket)

    def find_mbcd(self, x=-1):
        causal_matrix = self.causal_matrix
        num_var = causal_matrix.shape[0]
        mbcd = []
        for i in range(num_var):
            if causal_matrix[i, x] != 0:
                mbcd.append(i)
        if not mbcd:
            return []
        else:
            mbcd_set = set(mbcd)
            if num_var - 1 in mbcd_set:
                final_mbcd = list(mbcd_set.discard(num_var - 1))
            else:
                final_mbcd = list(mbcd_set)
            return final_mbcd

    def show_causal_graph(self):
        """
        Show the causal graph and save to the save path.
        :param causal_graph: The direct causal graph.
        :param save_path: The save path.
        :return:
        """
        save_path = self.save_path
        graphviz_dag = GraphUtils.to_pgv(self.causal_graph)
        graphviz_dag.draw(save_path, prog='dot', format='png')
        img = mpimg.imread(save_path)
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def mb_to_causal_matrix_target(self, mb, num_variables, target_variable=-1):
        causal_matrix = np.zeros([num_variables, num_variables])
        for i in mb:
            causal_matrix[i, target_variable] = 1

        return causal_matrix

    def weighted_dfs(self, matrix, current, target, visited, path, all_paths):
        if current == target:
            all_paths.append(path.copy())
            return

        visited[current] = True
        path.append(current)

        for i, weight in enumerate(matrix[current]):
            if weight != 0 and not visited[i]:
                self.weighted_dfs(matrix, i, target, visited, path, all_paths)

        path.pop()
        visited[current] = False

    def all_causal_paths_idetification(self, target_variable=-1):
        causal_matrix = self.causal_matrix
        if target_variable == -1:
            target_variable = causal_matrix.shape[0] - 1
        all_paths_to_target = []
        num_variables = causal_matrix.shape[0]
        for start_node in range(num_variables):
            if start_node != target_variable:
                self.weighted_dfs(causal_matrix, start_node, target_variable, [False] * num_variables, [], all_paths_to_target)

class CausalDiscovery:
    def show_causal_graph(self, causal_graph, save_path='test.png'):
        """
        Show the causal graph and save to the save path.
        :param causal_graph: The direct causal graph.
        :param save_path: The save path.
        :return:
        """
        graphviz_dag = GraphUtils.to_pgv(causal_graph)
        graphviz_dag.draw(save_path, prog='dot', format='png')
        img = mpimg.imread(save_path)
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def mb_to_causal_matrix(self, mb, num_variables):
        causal_matrix = np.zeros([num_variables, num_variables])
        for i in mb:
            causal_matrix[i, -1] = 1
        return causal_matrix

    def mb_to_causal_matrix_target(self, mb, num_variables, target_variable):
        causal_matrix = np.zeros([num_variables, num_variables])
        for i in mb:
            causal_matrix[i, target_variable] = 1

        return causal_matrix

    def cg_matrix_to_adjacent_matrix(self, A):
        """
        Convert causal graph matrix where A[i,j] == -1 and A[j,i] == 1, then i->j to
        a matrix B where B[i,j] == 1 then i->j, otherwise 0.
        :param A: The causal graph matrix
        :return: B
        """
        B = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] == -1 and A[j, i] == 1:
                    B[i, j] = 1
        return B

    def create_dag_from_matrix(self, causal_matrix, node_names_list):
        nodes = []
        causal_matrix
        for name in node_names_list:
            node = GraphNode(name)
            nodes.append(node)

        dag = Dag(nodes)

        num_variables = causal_matrix.shape[0]

        for i in range(num_variables):
            for j in range(num_variables):
                if causal_matrix[i, j] != 0:
                    dag.add_directed_edge(nodes[i], nodes[j])

        return dag

    def weighted_causal_matrix_to_adjacent_matrix(self, weighted_causal_matrix):
        B = np.zeros_like(weighted_causal_matrix)
        for i in range(weighted_causal_matrix.shape[0]):
            for j in range(weighted_causal_matrix.shape[1]):
                if weighted_causal_matrix[i, j] != 0:
                    B[i, j] = 1
        return B

    def creat_random_weighted_causal_matrix(self, causal_matrix):
        random_weighted_causal_matrix = copy.deepcopy(causal_matrix)
        num_variables = random_weighted_causal_matrix.shape[0]
        for i in range(num_variables):
            for j in range(num_variables):
                if causal_matrix[i, j] == 1:
                    random_weighted_causal_matrix[i, j] = random.uniform(0, 1)
        return random_weighted_causal_matrix

    def get_node_names_list(self, data_set_path):

        with open(data_set_path, 'r', newline='', encoding='utf-8-sig') as file:
            csv_reader = csv.reader(file)
            node_names = next(csv_reader)
            node_names_list = list(map(str, node_names))

        return node_names_list

    def is_subpath(self, path1, path2):
        '''
        determine whether path1 is a subpath of path2
        :param path1:
        :param path2:
        :return:
        '''
        test_path1 = copy.deepcopy(path1)
        test_path2 = copy.deepcopy(path2)
        test_path1.append(-1)
        test_path2.append(-1)
        str_path1 = '-'.join(map(str, test_path1))
        str_path2 = '-'.join(map(str, test_path2))

        return str_path1 in str_path2

    def is_overlapping(self, path1, path2):
        return any(node in path1 for node in path2)

    def show_all_path(self, all_paths, node_names_list):
        '''
        Print all paths of the target variable
        :param all_paths: the causal paths list
        :param node_names_list: the node name list
        :return:
        '''
        for path in all_paths:
            print(" -> ".join(map(str, [node_names_list[p] for p in path])))

    def dfs(self, matrix, node, target, visited, path, all_paths):
        '''
        Deep-first search, find the causal path of a given variable
        '''
        visited[node] = True
        path.append(node)

        if node == target:
            all_paths.append(path.copy())
        else:
            for i, val in enumerate(matrix[node]):
                if val == 1 and not visited[i]:
                    self.dfs(matrix, i, target, visited, path, all_paths)

        path.pop()
        visited[node] = False

    def causal_paths_identification(self, causal_graph_matrix, target_variable=-1):
        '''
        Find all causal path of the target variable
        :param target_variable: the index of the target variable
        :param causal_graph_matrix: the causal graph matrix of the target variable
        :return: the list of all causal paths.
        '''
        if target_variable == -1:
            target_variable = causal_graph_matrix.shape[0] - 1
        all_paths_to_target = []
        num_variables = causal_graph_matrix.shape[0]
        for start_node in range(num_variables):
            if start_node != target_variable:
                self.dfs(causal_graph_matrix, start_node, target_variable, [False] * num_variables, [], all_paths_to_target)

        final_paths = []
        for path in all_paths_to_target:
            # 如果新路径不是final_paths中的子路径
            if not any(self.is_subpath(path, p) for p in final_paths):
                # 删除final_paths中的所有路径，它们是新路径的子路径
                final_paths = [p for p in final_paths if not self.is_subpath(p, path)]
                # 添加新路径到final_paths
                final_paths.append(path)
        return final_paths

    def weighted_dfs(self, matrix, current, target, visited, path, all_paths):
        if current == target:
            all_paths.append(path.copy())
            return

        visited[current] = True
        path.append(current)

        for i, weight in enumerate(matrix[current]):
            if weight != 0 and not visited[i]:
                self.weighted_dfs(matrix, i, target, visited, path, all_paths)

        path.pop()
        visited[current] = False

    def weighted_causal_paths_identification(self, weighted_causal_graph_matrix, target_variable=-1):
        '''
        Find all causal path of the target variable
        :param target_variable: the index of the target variable
        :param causal_graph_matrix: the causal graph matrix of the target variable
        :return: the list of all causal paths.
        '''
        if target_variable == -1:
            target_variable = weighted_causal_graph_matrix.shape[0] - 1
        all_paths_to_target = []
        num_variables = weighted_causal_graph_matrix.shape[0]
        for start_node in range(num_variables):
            if start_node != target_variable:
                self.weighted_dfs(weighted_causal_graph_matrix, start_node, target_variable, [False] * num_variables, [], all_paths_to_target)

        # final_paths = self.remove_subpaths(all_paths_to_target)

        final_paths = []
        for path in all_paths_to_target:
            if not any(self.is_subpath(path, p) for p in final_paths):
                final_paths = [fp for fp in final_paths if not self.is_subpath(fp, path)]
                final_paths.append(path)

        return final_paths

    def covert_original_order_to_causal_order(self, node_names_list, causal_order):
        ordered_node_names_list = []
        for i in causal_order:
            ordered_node_names_list.append(node_names_list[i])
        return ordered_node_names_list

    def normalised_causal_matrix(self, causal_matrix):
        '''
        Normalised the obtained causal matrix.
        :param causal_matrix: weighted causal matrix
        :return:
        '''
        max_weight = np.max(causal_matrix)
        min_weight = np.min(causal_matrix)
        num_variables = causal_matrix.shape[0]
        normalised_causal_matrix = copy.deepcopy(causal_matrix)
        for i in range(num_variables):
            for j in range(num_variables):
                if causal_matrix[i, j] != 0:
                    normalised_causal_matrix[i, j] = (causal_matrix[i, j] - min_weight) / (max_weight-min_weight)
        return normalised_causal_matrix

    def calculate_all_causal_path_weights(self, all_causal_path, weighted_causal_matrix):
        all_causal_path_weight = []
        for causal_path in all_causal_path:
            path_weight = calculate_causal_weights(causal_path, weighted_causal_matrix)
            all_causal_path_weight.append(path_weight)
        return all_causal_path_weight

    def get_gt_sachs_causal_graph(self, sachs_data_set_path):

        with open(sachs_data_set_path, 'r', newline='', encoding='utf-8-sig') as file:
            csv_reader = csv.reader(file)
            node_names = next(csv_reader)
            node_names_list = list(map(str, node_names))

        nodes_list = []
        print(node_names_list)

        for name in node_names:
            node = GraphNode(name)
            nodes_list.append(node)
        dag = Dag(nodes_list)
        node = {}
        for name in node_names:
            node[name] = dag.get_node(name)

        dag.add_directed_edge(node['plc'], node['pip3'])
        dag.add_directed_edge(node['plc'], node['pip2'])
        dag.add_directed_edge(node['plc'], node['pkc'])
        dag.add_directed_edge(node['pip3'], node['pip2'])
        dag.add_directed_edge(node['pip3'], node['akt'])
        dag.add_directed_edge(node['pip2'], node['pkc'])
        dag.add_directed_edge(node['pkc'], node['mek'])
        dag.add_directed_edge(node['pkc'], node['raf'])
        dag.add_directed_edge(node['pkc'], node['pka'])
        dag.add_directed_edge(node['pkc'], node['jnk'])
        dag.add_directed_edge(node['pkc'], node['p38'])
        dag.add_directed_edge(node['pka'], node['raf'])
        dag.add_directed_edge(node['pka'], node['mek'])
        dag.add_directed_edge(node['pka'], node['erk'])
        dag.add_directed_edge(node['pka'], node['akt'])
        dag.add_directed_edge(node['pka'], node['jnk'])
        dag.add_directed_edge(node['pka'], node['p38'])
        dag.add_directed_edge(node['raf'], node['mek'])
        dag.add_directed_edge(node['mek'], node['erk'])
        dag.add_directed_edge(node['erk'], node['akt'])

        return dag, nodes_list

    def evaluate_causal_matrix(self, ground_truth_causal_matrix, predicted_causal_matrix):
        flat_ground_truth = ground_truth_causal_matrix.ravel()
        flat_predicted = predicted_causal_matrix.ravel()

        accuracy = accuracy_score(flat_ground_truth, flat_predicted)
        precision = precision_score(flat_ground_truth, flat_predicted)
        recall = recall_score(flat_ground_truth, flat_predicted)
        f1 = f1_score(flat_ground_truth, flat_predicted)
        return accuracy, precision, recall, f1


def find_direct_cause(causal_matrix, x=-1):
    num_var = causal_matrix.shape[0]
    mbcd = []
    for i in range(num_var):
        if causal_matrix[i, x] != 0:
            mbcd.append(i)
    if not mbcd:
        return []
    else:
        mbcd_set = set(mbcd)
        if num_var-1 in mbcd_set:
            final_mbcd = list(mbcd_set.discard(num_var-1))
        else:
            final_mbcd = list(mbcd_set)
        return final_mbcd

def find_markov_blanket(causal_matrix, x=-1):
    # 获取矩阵的维度
    num_var = causal_matrix.shape[0]
    # 1. 获取X的所有父节点
    parents_of_X = [i for i in range(num_var) if causal_matrix[i, x] != 0]

    # 2. 获取X的所有子节点
    children_of_X = [i for i in range(num_var) if causal_matrix[x, i] != 0]

    # 3. 获取X的所有子节点的其他父节点
    spouses_of_X = []
    for child in children_of_X:
        spouses_of_X.extend([i for i in range(num_var) if causal_matrix[i, child] != 0 and i != x])

    # 合并所有的组件并去除重复的以及目标变量，即最后一个变量
    markov_blanket = set(parents_of_X + children_of_X + spouses_of_X)
    markov_blanket.discard(num_var - 1)

    return list(markov_blanket)

def encode_labels(df):
    '''
    Covert string labels to discrete labels.
    :param df:
    :return:
    '''
    le = LabelEncoder()
    ledf = copy.deepcopy(df)
    ledf.iloc[:, -1] = le.fit_transform(ledf.iloc[:, -1])
    return ledf, le.classes_


def calculate_causal_weights(causal_path, causal_matrix):
    path_weights = 1
    num_edges = len(causal_path) - 1  # number of edges of the given causal path
    for i in range(num_edges):
        path_weights *= causal_matrix[causal_path[i], causal_path[i+1]]
    path_weights = abs(path_weights)
    return path_weights


def create_classification_data_from_regression_data(regression_data, target_variable=-1):
    '''
    Create a classification data set from a regression data set.
    :param regression_data: The original regression data set
    :param target_variable: The variable to be coverted to discrete
    :return: The classification data
    '''

    classification_data = copy.deepcopy(regression_data)

    a = np.percentile(classification_data[:, target_variable], 50.00)
    # b = np.percentile(classification_data[:, target_variable], 66.67)

    # 对第d维进行替换
    # classification_data[:, target_variable] = np.where(classification_data[:, target_variable] < a, 0,
    #                                                np.where(classification_data[:, target_variable] < b, 1, 2))
    classification_data[:, target_variable] = np.where(classification_data[:, target_variable] < a, 0, 1)

    # if target_variable != -1:
    #     classification_data = np.hstack((np.delete(classification_data, target_variable, axis=1),
    #                                     classification_data[:, target_variable][:, np.newaxis]))

    return classification_data

def create_moved_node_names_list(node_names_list, moved_variable_index):
    moved_feature = node_names_list.pop(moved_variable_index)
    node_names_list.append(moved_feature)

    return node_names_list