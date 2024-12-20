import io
import copy
import csv
import random
import json

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import skfuzzy as fuzz
import matplotlib.pyplot as plt


from causallearn.graph.Dag import Dag
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


from collections import defaultdict

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

    def create_dag_from_matrix(self, causal_matrix, node_names_list):
        nodes = []
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

    # def calculate_all_causal_path_weights(self, all_causal_path, weighted_causal_matrix):
    #     all_causal_path_weight = []
    #     for causal_path in all_causal_path:
    #         path_weight = calculate_causal_weights(causal_path, weighted_causal_matrix)
    #         all_causal_path_weight.append(path_weight)
    #     return all_causal_path_weight

    def calculate_all_causal_path_weights_product(self, all_causal_path, weighted_causal_matrix):
        all_causal_path_weight = []
        target_variable_index = weighted_causal_matrix.shape[0]-1
        for causal_path in all_causal_path:
            causal_path.append(target_variable_index)
            path_weights = 1
            num_edges = len(causal_path) - 1  # number of edges of the given causal path
            for i in range(num_edges):
                path_weights *= weighted_causal_matrix[causal_path[i], causal_path[i + 1]]
            path_weights = abs(path_weights)
            all_causal_path_weight.append(path_weights)
        return all_causal_path_weight


    def calculate_all_causal_path_weights_add(self, all_causal_path, weighted_causal_matrix):
        all_causal_path_weight = []
        target_variable_index = weighted_causal_matrix.shape[0] - 1
        for causal_path in all_causal_path:
            path_weights = 0
            num_edges = len(causal_path) - 1  # number of edges of the given causal path
            for i in range(num_edges):
                path_weights += abs(weighted_causal_matrix[causal_path[i], causal_path[i + 1]])
            all_causal_path_weight.append(path_weights)
        return all_causal_path_weight


    def calculate_all_causal_path_weights_weighted(self, all_causal_path, weighted_causal_matrix):
        all_causal_path_weight = []
        target_variable_index = weighted_causal_matrix.shape[0] - 1
        for causal_path in all_causal_path:
            path_weights = 0
            num_edges = len(causal_path) - 1  # number of edges of the given causal path
            for i in range(num_edges):
                weights_coefficient = 1/(num_edges-i)
                weighted_weights = weights_coefficient * weighted_causal_matrix[causal_path[i], causal_path[i + 1]]
                path_weights += abs(weighted_weights)
            all_causal_path_weight.append(path_weights)
        return all_causal_path_weight

    def get_gt_sachs_causal_graph(self, sachs_data_set_path):

        with open(sachs_data_set_path, 'r', newline='', encoding='utf-8-sig') as file:
            csv_reader = csv.reader(file)
            node_names = next(csv_reader)
            node_names_list = list(map(str, node_names))

        nodes_list = []
        # print(node_names_list)

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


class RuleGeneration:
    def define_fuzzy_set_for_each_normalised_variable(self, centers, feature_range=np.linspace(0, 1, 1000)):
        '''
        Define fuzzy sets in range [0, 1]
        :param feature_range: np.linspace(0, 1, 1000), i.e., 0-1范围内进行1000等分
        :param centers: cluster centers of each variable.
        :return: a list of fuzzy set
        '''
        centers = np.abs(centers)
        a = np.min(centers)
        b = np.median(centers)
        c = np.max(centers)
        if a < 0:
            a = 0
        if c > 1:
            c = 1
        low = fuzz.trapmf(feature_range,
                          [0, 0, a, b])
        mid = fuzz.trimf(feature_range,
                         [a, b, c])
        high = fuzz.trapmf(feature_range,
                           [b, c, 1, 1])
        return low, mid, high

    def define_fuzzy_sets_for_all_normalised_variable(self, normalised_x, num_mf=3, variable_range=np.linspace(0, 1, 1000)):
        num_variables = normalised_x.shape[1]
        variables_fuzzy_sets = list(range(num_variables))
        for i in range(num_variables):
            current_x = copy.deepcopy(normalised_x[:, i])
            reshaped_current_x = current_x.reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, init='random', n_init=50, max_iter=5000, tol=1e-6, algorithm='elkan')
            kmeans.fit(reshaped_current_x)
            centers = kmeans.cluster_centers_
            sorted_centers = np.sort(centers, axis=0)
            variables_fuzzy_sets[i] = self.define_fuzzy_set_for_each_normalised_variable(sorted_centers, variable_range)

        return variables_fuzzy_sets

    def wang_mendel_algorithm(self, x, y, num_mf=3, variable_range=np.linspace(0, 1, 1000)):
        rules = {}
        fuzzy_sets = self.define_fuzzy_sets_for_all_normalised_variable(x, num_mf, variable_range)
        for idx, sample in enumerate(x):
            rule_key = ()
            max_md = 0
            for i, feature_value in enumerate(sample):
                memberships = [
                    (fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value), "Low"),
                    (fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value), "Mid"),
                    (fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value), "High")
                ]

                antecedent = max(memberships, key=lambda item: item[0])[1]
                rule_key += (antecedent,)

            if rule_key not in rules or max_md < max([md[0] for md in memberships]):
                rules[rule_key] = y[idx]

        return rules, fuzzy_sets

    def mablar_cw_wm_train(self, original_x, y, all_causal_paths, num_mf=3, variable_range=np.linspace(0, 1, 1000)):
        '''

        :param original_x:the normalised original x, i.e., unordered and including the target variable data set.
        :param y:
        :param all_causal_paths:
        :param weighted_causal_matrix:
        :param num_mf:
        :param variable_range:
        :return:
        '''
        fuzzy_system_number = 0
        all_fuzzy_systems = defaultdict(dict)
        all_fuzzy_sets = defaultdict(dict)
        for variable_set in all_causal_paths:
            train_x = original_x[:, variable_set]
            all_fuzzy_systems[fuzzy_system_number], all_fuzzy_sets[fuzzy_system_number] = self.wang_mendel_algorithm(train_x, y, num_mf, variable_range)
            fuzzy_system_number += 1
        return all_fuzzy_systems, all_fuzzy_sets

    def mablar_cw_wm_predict(self, sample, all_causal_path, all_fuzzy_systems, all_fuzzy_sets, all_weights):
        num_fuzzy_systems = len(all_causal_path)
        all_predicts = []
        for i in range(num_fuzzy_systems):
            current_variable_set = all_causal_path[i]
            current_variable_set = np.array(current_variable_set)
            current_input = sample[current_variable_set]
            current_fuzzy_system = all_fuzzy_systems[i]
            current_fuzzy_sets = all_fuzzy_sets[i]
            current_pre = self.wm_predict(current_input, current_fuzzy_system, current_fuzzy_sets)
            all_predicts.append(current_pre)
        final_predict = self.weighted_voting(all_predicts, all_weights)

        return final_predict


    def wm_predict(self, sample, rules, fuzzy_sets, variable_range=np.linspace(0, 1, 1000)):
        max_match_degree = -np.inf
        predicted_class = None

        for rule_key, rule_class in rules.items():
            match_degree = 1
            for i, feature_value in enumerate(sample):
                if rule_key[i] == "Low":
                    membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value)
                elif rule_key[i] == "Mid":
                    membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value)
                elif rule_key[i] == "High":
                    membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value)

                match_degree *= membership

            if match_degree > max_match_degree:
                max_match_degree = match_degree
                predicted_class = rule_class

        return predicted_class

    def get_best_matching_rule(self, sample, rules, fuzzy_sets, variable_range=np.linspace(0, 1, 1000)):
        max_match_degree = -np.inf
        predicted_class = None
        best_rule = None

        for rule_key, rule_class in rules.items():
            match_degree = 1
            for i, feature_value in enumerate(sample):
                if rule_key[i] == "Low":
                    membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value)
                elif rule_key[i] == "Mid":
                    membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value)
                elif rule_key[i] == "High":
                    membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value)

                match_degree *= membership

            if match_degree > max_match_degree:
                max_match_degree = match_degree
                predicted_class = rule_class
                best_rule = rule_key, rule_class

        return predicted_class, best_rule, max_match_degree

    def get_best_mbcw_matching_rules(self, cla_normalised_data, test_sample_index, all_fuzzy_systems, all_fuzzy_sets, all_causal_path, all_causal_weights):
        num_fuzzy_systems = len(all_causal_path)
        weighted_rules_list = []
        unsorted_causal_path = []
        for i in range(num_fuzzy_systems):
            current_path = all_causal_path[i]
            current_fuzzy_system = all_fuzzy_systems[i]
            current_fuzzy_set = all_fuzzy_sets[i]
            current_causal_weight = all_causal_weights[i]
            current_sample = cla_normalised_data[test_sample_index, current_path]
            predicted_class, best_rule, max_match_degree = self.get_best_matching_rule(current_sample, current_fuzzy_system, current_fuzzy_set)
            rule_weights = max_match_degree*current_causal_weight
            current_weighted_rule = best_rule + (rule_weights,)
            current_path.append(rule_weights)
            unsorted_causal_path.append(current_path)
            weighted_rules_list.append(current_weighted_rule)
        sorted_weighted_rules_list = sorted(weighted_rules_list, key=lambda X:X[2], reverse=True)
        sorted_causal_path = sorted(unsorted_causal_path, key=lambda X:X[-1], reverse=True)
        return sorted_weighted_rules_list,sorted_causal_path

    def wm_cross_validation(self, normalised_X, original_Y):
        '''
        Test the obtained rules using cross_validation
        :param normalised_data: a numpy matrix
        :return:
        '''
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        fold_results = {}

        variable_range = np.linspace(0, 1, 1000)

        for fold_idx, (train_index, test_index) in enumerate(kf.split(normalised_X)):
            # 划分训练集和测试集
            X_train, X_test = normalised_X[train_index], normalised_X[test_index]
            y_train, y_test = original_Y[train_index], original_Y[test_index]

            fold_name = f'fold_{fold_idx + 1}'
            fold_results[fold_name] = {}

            # 使用Wang-Mendel算法从训练数据中提取规则
            rule_base, variables_fuzzy_sets = self.wang_mendel_algorithm(X_train, y_train)

            # 使用规则对测试数据进行分类
            predictions = [self.wm_predict(sample, rule_base, variables_fuzzy_sets, variable_range) for sample in
                           X_test]

            # 计算并存储每次的准确率
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)
            print("Fold accuracy:", accuracy)
            fold_results[fold_name]['rule_base'] = rule_base
            fold_results[fold_name]['accuracy'] = accuracy

        # 计算平均准确率
        average_accuracy = np.mean(accuracies)
        print("Average accuracy:", average_accuracy)
        return fold_results, average_accuracy

    def mablar_cw_wm_cross_validation(self, normalised_X, original_Y, all_causal_path, all_causal_weights):
        '''
        Test the obtained rules using cross_validation
        :param normalised_data: a numpy matrix
        :return:
        '''

        kf = KFold(n_splits=5, shuffle=True, random_state=4)
        accuracies = []
        fold_results = {}

        for fold_idx, (train_index, test_index) in enumerate(kf.split(normalised_X)):
            # 划分训练集和测试集
            X_train, X_test = normalised_X[train_index], normalised_X[test_index]
            y_train, y_test = original_Y[train_index], original_Y[test_index]

            fold_name = f'fold_{fold_idx + 1}'
            fold_results[fold_name] = {}

            # 使用Wang-Mendel算法从训练数据中提取规则
            all_fuzzy_systems, all_fuzzy_sets = self.mablar_cw_wm_train(X_train, y_train, all_causal_path)
            # 使用规则对测试数据进行分类
            predictions = [self.mablar_cw_wm_predict(sample, all_causal_path, all_fuzzy_systems, all_fuzzy_sets, all_causal_weights) for sample in
                           X_test]

            # 计算并存储每次的准确率
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)
            # print("Fold accuracy:", accuracy)
            fold_results[fold_name]['rule_base'] = all_fuzzy_systems
            fold_results[fold_name]['accuracy'] = accuracy

        # 计算平均准确率
        average_accuracy = np.mean(accuracies)
        # print("Average accuracy:", average_accuracy)
        return fold_results, average_accuracy

    def weighted_voting(self, classifiers_output, weights):
        if len(classifiers_output) != len(weights):
            raise ValueError("The number of classifiers' outputs and weights should be the same")

        # 初始化得票数为0
        votes = defaultdict(float)

        # 为每个类别计算得票数
        for cls_output, weight in zip(classifiers_output, weights):
            votes[cls_output] += weight

        # 返回得票数最多的类别
        return max(votes, key=votes.get)

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