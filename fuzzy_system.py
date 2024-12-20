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

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from collections import defaultdict


class FuzzySystem:
    def __init__(self, rule_base=None, fuzzy_sets=None, variable_range=np.linspace(0, 1, 1000)):
        self.rule_base = rule_base
        self.fuzzy_sets = fuzzy_sets
        self.variable_range = variable_range

    def fuzzy_sets_definition(self, x, clu_paras=None):
        num_variables = x.shape[1]
        variables_fuzzy_sets = list(range(num_variables))
        variable_range = self.variable_range
        clus_num = clu_paras.clus_num
        clu_init = clu_paras.clu_init
        clu_alg = clu_paras.clu_alg

        # clus_num = 3
        # clu_init = 'random'
        # clu_alg = 'elkan'

        for i in range(num_variables):
            current_x = copy.deepcopy(x[:, i])
            reshaped_current_x = current_x.reshape(-1, 1)
            kmeans = KMeans(n_clusters=clus_num, init=clu_init, n_init=50, max_iter=5000, tol=1e-6, algorithm=clu_alg)
            kmeans.fit(reshaped_current_x)
            current_centers = kmeans.cluster_centers_
            # sorted_centers = np.sort(centers, axis=0)
            a = np.min(current_centers)
            b = np.median(current_centers)
            c = np.max(current_centers)
            if a < 0:
                a = 0
            if c > 1:
                c = 1
            low = fuzz.trapmf(variable_range,
                              [0, 0, a, b])
            mid = fuzz.trimf(variable_range,
                             [a, b, c])
            high = fuzz.trapmf(variable_range,
                               [b, c, 1, 1])
            variables_fuzzy_sets[i] = (low, mid, high)

        return variables_fuzzy_sets

    def fit_wm(self, x, y, clu_paras):
        rules = {}
        fuzzy_sets = self.fuzzy_sets_definition(x, clu_paras)
        variable_range = self.variable_range
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
        self.rule_base = rules
        self.fuzzy_sets = fuzzy_sets

    def predict_all_rules(self, sample):
        max_match_degree = 0
        predicted_class = None
        best_rule = None

        rules = self.rule_base
        fuzzy_sets = self.fuzzy_sets
        variable_range = self.variable_range
        labels = set(rules.values())

        for current_class in labels:
            current_class_match_degree = 0
            current_max_rule_match_degree = -np.inf
            for rule_key, rule_class in rules.items():
                if rule_class == current_class:
                    match_degree = 1
                    for i, feature_value in enumerate(sample):
                        if rule_key[i] == "Low":
                            membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value)
                        elif rule_key[i] == "Mid":
                            membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value)
                        elif rule_key[i] == "High":
                            membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value)
                        match_degree *= membership
                    if match_degree >= current_max_rule_match_degree:
                        current_best_rule = rule_key, rule_class
                        current_max_rule_match_degree = match_degree

                    current_class_match_degree += match_degree

            if current_class_match_degree >= max_match_degree:
                max_match_degree = current_class_match_degree
                predicted_class = current_class
                best_rule = current_best_rule
                max_rule_match_degree = current_max_rule_match_degree

        return predicted_class, best_rule, max_rule_match_degree


    def predict_all_rules_mbcw(self, sample, causal_weight):
        max_match_degree = 0
        predicted_class = None
        best_rule = None

        rules = self.rule_base
        fuzzy_sets = self.fuzzy_sets
        variable_range = self.variable_range
        labels = set(rules.values())
        fs_dict = defaultdict(float)

        for current_class in labels:
            current_class_match_degree = 0
            current_max_rule_match_degree = -np.inf
            for rule_key, rule_class in rules.items():
                if rule_class == current_class:
                    match_degree = 1
                    for i, feature_value in enumerate(sample):
                        if rule_key[i] == "Low":
                            membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value)
                        elif rule_key[i] == "Mid":
                            membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value)
                        elif rule_key[i] == "High":
                            membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value)
                        match_degree *= membership
                    if match_degree >= current_max_rule_match_degree:
                        current_best_rule = rule_key, rule_class
                        current_max_rule_match_degree = match_degree

                    current_class_match_degree += match_degree
            fs_dict[current_class] = current_class_match_degree*causal_weight

            # if current_class_match_degree >= max_match_degree:
            #     max_match_degree = current_class_match_degree
            #     predicted_class = current_class
            #     best_rule = current_best_rule
            #     max_rule_match_degree = current_max_rule_match_degree

        return fs_dict


    def predict(self, sample):
        max_match_degree = -np.inf
        predicted_class = None
        best_rule = None

        rules = self.rule_base
        fuzzy_sets = self.fuzzy_sets
        variable_range = self.variable_range

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


# class RuleGeneration:
#
#     def define_fuzzy_set_for_each_normalised_variable(self, centers, feature_range=np.linspace(0, 1, 1000)):
#         '''
#         Define fuzzy sets in range [0, 1]
#         :param feature_range: np.linspace(0, 1, 1000), i.e., 0-1范围内进行1000等分
#         :param centers: cluster centers of each variable.
#         :return: a list of fuzzy set
#         '''
#         centers = np.abs(centers)
#         a = np.min(centers)
#         b = np.median(centers)
#         c = np.max(centers)
#         if c > 1:
#             c = 1
#         low = fuzz.trapmf(feature_range,
#                           [0, 0, a, b])
#         mid = fuzz.trimf(feature_range,
#                          [a, b, c])
#         high = fuzz.trapmf(feature_range,
#                            [b, c, 1, 1])
#         return low, mid, high
#
#     def define_fuzzy_sets_for_all_normalised_variable(self, normalised_x, num_mf=3,
#                                                       variable_range=np.linspace(0, 1, 1000)):
#         num_variables = normalised_x.shape[1]
#         variables_fuzzy_sets = list(range(num_variables))
#         for i in range(num_variables):
#             current_x = copy.deepcopy(normalised_x[:, i])
#             reshaped_current_x = current_x.reshape(-1, 1)
#             kmeans = KMeans(n_clusters=num_mf, n_init=10)
#             kmeans.fit(reshaped_current_x)
#             centers = kmeans.cluster_centers_
#             sorted_centers = np.sort(centers, axis=0)
#             variables_fuzzy_sets[i] = self.define_fuzzy_set_for_each_normalised_variable(sorted_centers, variable_range)
#
#         return variables_fuzzy_sets
#
#     def wang_mendel_algorithm(self, x, y, num_mf=3, variable_range=np.linspace(0, 1, 1000)):
#         rules = {}
#         fuzzy_sets = self.define_fuzzy_sets_for_all_normalised_variable(x, num_mf, variable_range)
#         for idx, sample in enumerate(x):
#             rule_key = ()
#             max_md = 0
#             for i, feature_value in enumerate(sample):
#                 memberships = [
#                     (fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value), "Low"),
#                     (fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value), "Mid"),
#                     (fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value), "High")
#                 ]
#
#                 antecedent = max(memberships, key=lambda item: item[0])[1]
#                 rule_key += (antecedent,)
#
#             if rule_key not in rules or max_md < max([md[0] for md in memberships]):
#                 rules[rule_key] = y[idx]
#
#         return rules, fuzzy_sets
#
#     def mablar_cw_wm_train(self, original_x, y, all_causal_paths, num_mf=3, variable_range=np.linspace(0, 1, 1000)):
#         '''
#
#         :param original_x:the normalised original x, i.e., unordered and including the target variable data set.
#         :param y:
#         :param all_causal_paths:
#         :param weighted_causal_matrix:
#         :param num_mf:
#         :param variable_range:
#         :return:
#         '''
#         fuzzy_system_number = 0
#         all_fuzzy_systems = defaultdict(dict)
#         all_fuzzy_sets = defaultdict(dict)
#         for variable_set in all_causal_paths:
#             train_x = original_x[:, variable_set]
#             all_fuzzy_systems[fuzzy_system_number], all_fuzzy_sets[fuzzy_system_number] = self.wang_mendel_algorithm(
#                 train_x, y, num_mf, variable_range)
#             fuzzy_system_number += 1
#         return all_fuzzy_systems, all_fuzzy_sets
#
#     def mablar_cw_wm_predict(self, sample, all_causal_path, all_fuzzy_systems, all_fuzzy_sets, all_weights):
#         num_fuzzy_systems = len(all_causal_path)
#         all_predicts = []
#         for i in range(num_fuzzy_systems):
#             current_variable_set = all_causal_path[i]
#             current_variable_set = np.array(current_variable_set)
#             current_input = sample[current_variable_set]
#             current_fuzzy_system = all_fuzzy_systems[i]
#             current_fuzzy_sets = all_fuzzy_sets[i]
#             current_pre = self.wm_predict(current_input, current_fuzzy_system, current_fuzzy_sets)
#             all_predicts.append(current_pre)
#         final_predict = self.weighted_voting(all_predicts, all_weights)
#
#         return final_predict
#
#     def wm_predict(self, sample, rules, fuzzy_sets, variable_range=np.linspace(0, 1, 1000)):
#         max_match_degree = -np.inf
#         predicted_class = None
#
#         for rule_key, rule_class in rules.items():
#             match_degree = 1
#             for i, feature_value in enumerate(sample):
#                 if rule_key[i] == "Low":
#                     membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value)
#                 elif rule_key[i] == "Mid":
#                     membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value)
#                 elif rule_key[i] == "High":
#                     membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value)
#
#                 match_degree *= membership
#
#             if match_degree > max_match_degree:
#                 max_match_degree = match_degree
#                 predicted_class = rule_class
#
#         return predicted_class
#
#     def get_best_matching_rule(self, sample, rules, fuzzy_sets, variable_range=np.linspace(0, 1, 1000)):
#         max_match_degree = -np.inf
#         predicted_class = None
#         best_rule = None
#
#         for rule_key, rule_class in rules.items():
#             match_degree = 1
#             for i, feature_value in enumerate(sample):
#                 if rule_key[i] == "Low":
#                     membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value)
#                 elif rule_key[i] == "Mid":
#                     membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value)
#                 elif rule_key[i] == "High":
#                     membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value)
#
#                 match_degree *= membership
#
#             if match_degree > max_match_degree:
#                 max_match_degree = match_degree
#                 predicted_class = rule_class
#                 best_rule = rule_key, rule_class
#
#         return predicted_class, best_rule, max_match_degree
#
#     def get_best_mbcw_matching_rules(self, cla_normalised_data, test_sample_index, all_fuzzy_systems, all_fuzzy_sets,
#                                      all_causal_path, all_causal_weights):
#         num_fuzzy_systems = len(all_causal_path)
#         weighted_rules_list = []
#         unsorted_causal_path = []
#         for i in range(num_fuzzy_systems):
#             current_path = all_causal_path[i]
#             current_fuzzy_system = all_fuzzy_systems[i]
#             current_fuzzy_set = all_fuzzy_sets[i]
#             current_causal_weight = all_causal_weights[i]
#             current_sample = cla_normalised_data[test_sample_index, current_path]
#             predicted_class, best_rule, max_match_degree = self.get_best_matching_rule(current_sample,
#                                                                                        current_fuzzy_system,
#                                                                                        current_fuzzy_set)
#             rule_weights = max_match_degree * current_causal_weight
#             current_weighted_rule = best_rule + (rule_weights,)
#             current_path.append(rule_weights)
#             unsorted_causal_path.append(current_path)
#             weighted_rules_list.append(current_weighted_rule)
#         sorted_weighted_rules_list = sorted(weighted_rules_list, key=lambda X: X[2], reverse=True)
#         sorted_causal_path = sorted(unsorted_causal_path, key=lambda X: X[-1], reverse=True)
#         return sorted_weighted_rules_list, sorted_causal_path
#
#     def wm_cross_validation(self, normalised_X, original_Y):
#         '''
#         Test the obtained rules using cross_validation
#         :param normalised_data: a numpy matrix
#         :return:
#         '''
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)
#         accuracies = []
#         fold_results = {}
#
#         variable_range = np.linspace(0, 1, 1000)
#
#         for fold_idx, (train_index, test_index) in enumerate(kf.split(normalised_X)):
#             # 划分训练集和测试集
#             X_train, X_test = normalised_X[train_index], normalised_X[test_index]
#             y_train, y_test = original_Y[train_index], original_Y[test_index]
#
#             fold_name = f'fold_{fold_idx + 1}'
#             fold_results[fold_name] = {}
#
#             # 使用Wang-Mendel算法从训练数据中提取规则
#             rule_base, variables_fuzzy_sets = self.wang_mendel_algorithm(X_train, y_train)
#
#             # 使用规则对测试数据进行分类
#             predictions = [self.wm_predict(sample, rule_base, variables_fuzzy_sets, variable_range) for sample in
#                            X_test]
#
#             # 计算并存储每次的准确率
#             accuracy = accuracy_score(y_test, predictions)
#             accuracies.append(accuracy)
#             print("Fold accuracy:", accuracy)
#             fold_results[fold_name]['rule_base'] = rule_base
#             fold_results[fold_name]['accuracy'] = accuracy
#
#         # 计算平均准确率
#         average_accuracy = np.mean(accuracies)
#         print("Average accuracy:", average_accuracy)
#         return fold_results, average_accuracy
#
#     def mablar_cw_wm_cross_validation(self, normalised_X, original_Y, all_causal_path, all_causal_weights):
#         '''
#         Test the obtained rules using cross_validation
#         :param normalised_data: a numpy matrix
#         :return:
#         '''
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)
#         accuracies = []
#         fold_results = {}
#
#         for fold_idx, (train_index, test_index) in enumerate(kf.split(normalised_X)):
#             # 划分训练集和测试集
#             X_train, X_test = normalised_X[train_index], normalised_X[test_index]
#             y_train, y_test = original_Y[train_index], original_Y[test_index]
#
#             fold_name = f'fold_{fold_idx + 1}'
#             fold_results[fold_name] = {}
#
#             # 使用Wang-Mendel算法从训练数据中提取规则
#             all_fuzzy_systems, all_fuzzy_sets = self.mablar_cw_wm_train(X_train, y_train, all_causal_path)
#             # 使用规则对测试数据进行分类
#             predictions = [self.mablar_cw_wm_predict(sample, all_causal_path, all_fuzzy_systems, all_fuzzy_sets,
#                                                      all_causal_weights) for sample in
#                            X_test]
#
#             # 计算并存储每次的准确率
#             accuracy = accuracy_score(y_test, predictions)
#             accuracies.append(accuracy)
#             print("Fold accuracy:", accuracy)
#             fold_results[fold_name]['rule_base'] = all_fuzzy_systems
#             fold_results[fold_name]['accuracy'] = accuracy
#
#         # 计算平均准确率
#         average_accuracy = np.mean(accuracies)
#         print("Average accuracy:", average_accuracy)
#         return fold_results, average_accuracy
#
#     def weighted_voting(self, classifiers_output, weights):
#         if len(classifiers_output) != len(weights):
#             raise ValueError("The number of classifiers' outputs and weights should be the same")
#
#         # 初始化得票数为0
#         votes = defaultdict(float)
#
#         # 为每个类别计算得票数
#         for cls_output, weight in zip(classifiers_output, weights):
#             votes[cls_output] += weight
#
#         # 返回得票数最多的类别
#         return max(votes, key=votes.get)
