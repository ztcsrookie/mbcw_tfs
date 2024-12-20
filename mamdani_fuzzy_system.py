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
from membership_functions import *


class SimpleMamdaniFuzzySystem:
    def __init__(self, rule_base=None, linguictic_rule_base=None, fuzzy_sets_parameters=None):
        '''
        :param rule_base: numpy matrix, R*D, R: #rules, D:#variables
        :param fuzzy_sets_parameters: numpy matrix, P*D, P: #fuzzy sets of each variable. D:#variables
        '''
        self.rule_base = rule_base
        self.linguistic_rule_base = linguictic_rule_base
        self.fuzzy_sets_parameters = fuzzy_sets_parameters

    def predict(self, x) -> int:
        '''
        
        :param x: an input sample 
        :return: y: the predicted label
        '''
        n_rules = self.rule_base.shape[0]
        D = self.rule_base.shape[1] - 1
        md_rules = np.ones((n_rules, D))
        for r in range(n_rules):
            for d in range(D):
                # print(x[d])
                a = self.fuzzy_sets_parameters[0, d]
                b = self.fuzzy_sets_parameters[1, d]
                c = self.fuzzy_sets_parameters[2, d]
                if self.rule_base[r, d] == 0:
                    md_rules[r, d] = left_shoulder_mf(x[d], a, b)
                elif self.rule_base[r, d] == 1:
                    md_rules[r, d] = triangle_mf(x[d], a, b, c)
                else:
                    md_rules[r, d] = right_shoulder_mf(x[d], b, c)
        prod_md_rules = np.prod(md_rules, axis=1, keepdims=True)
        max_row_index = np.argmax(prod_md_rules)

        predict_y = self.rule_base[max_row_index,-1]
        return predict_y

    def wm_fit(self, x, y, n_clusters=3) ->None:
        '''

        :param x: the features, nparray N*D, N: #samples, D: #features
        :param y: the label. nparray N*1. N: #samples,
        :return:
        '''
        N, D = x.shape

        # u, cntr, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        #     x, n_clusters, 1.5, error=0.00001, maxiter=10000, init=None)

        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=20, max_iter=1000, tol=1e-6, algorithm='elkan')
        kmeans.fit(x)
        cntr = kmeans.cluster_centers_

        self.fuzzy_sets_parameters = np.sort(cntr, axis=0)
        rule_base_dict = {}
        linguistic_rule_base = {}
        md_dict = {}

        for idx, sample in enumerate(x):
            current_product_md = 1
            candidate_rule = ()
            candidate_linguistic_rule = ()
            for i, feature_value in enumerate(sample):
                a = self.fuzzy_sets_parameters[0, i]
                b = self.fuzzy_sets_parameters[1, i]
                c = self.fuzzy_sets_parameters[2, i]
                md_low = left_shoulder_mf(feature_value, a, b)
                md_medium = triangle_mf(feature_value, a, b, c)
                md_high = right_shoulder_mf(feature_value, b, c)
                memberships = [
                    (md_low, 0, "Low"),
                    (md_medium, 1, "Mid"),
                    (md_high, 2, "High")
                ]
                max_md_i = max([md[0] for md in memberships])
                current_product_md *= max_md_i
                antecedent = max(memberships, key=lambda item: item[0])[1]
                linguistic_antecedent = max(memberships, key=lambda item: item[0])[2]
                candidate_rule += (antecedent,)
                candidate_linguistic_rule += (linguistic_antecedent,)


            if candidate_linguistic_rule not in linguistic_rule_base:
                linguistic_rule_base[candidate_linguistic_rule] = y[idx]
                rule_base_dict[candidate_rule] = y[idx]
                md_dict[candidate_linguistic_rule] = current_product_md

            elif candidate_linguistic_rule in linguistic_rule_base and current_product_md > md_dict[candidate_linguistic_rule]:
                linguistic_rule_base[candidate_linguistic_rule] = y[idx]
                rule_base_dict[candidate_rule] = y[idx]
                md_dict[candidate_linguistic_rule] = current_product_md


        numeric_rule_base = np.zeros((len(rule_base_dict), D+1))
        r=0
        for key, value in rule_base_dict.items():
            for i, idx in enumerate(key):
                numeric_rule_base[r, i] = idx
            numeric_rule_base[r,-1] = value
            r += 1
        self.rule_base = numeric_rule_base
        self.linguistic_rule_base = linguistic_rule_base


#
# class FuzzySystem:
#     def __init__(self, rule_base=None, fuzzy_sets=None, variable_range=np.linspace(0, 1, 1000)):
#         self.rule_base = rule_base
#         self.fuzzy_sets = fuzzy_sets
#         self.variable_range = variable_range
#
#     def fuzzy_sets_definition(self, x, clu_paras=None):
#         num_variables = x.shape[1]
#         variables_fuzzy_sets = list(range(num_variables))
#         variable_range = self.variable_range
#         clus_num = clu_paras.clus_num
#         clu_init = clu_paras.clu_init
#         clu_alg = clu_paras.clu_alg
#
#         # clus_num = 3
#         # clu_init = 'random'
#         # clu_alg = 'elkan'
#
#         for i in range(num_variables):
#             current_x = copy.deepcopy(x[:, i])
#             reshaped_current_x = current_x.reshape(-1, 1)
#             kmeans = KMeans(n_clusters=clus_num, init=clu_init, n_init=50, max_iter=5000, tol=1e-6, algorithm=clu_alg)
#             kmeans.fit(reshaped_current_x)
#             current_centers = kmeans.cluster_centers_
#             # sorted_centers = np.sort(centers, axis=0)
#             a = np.min(current_centers)
#             b = np.median(current_centers)
#             c = np.max(current_centers)
#             if a < 0:
#                 a = 0
#             if c > 1:
#                 c = 1
#             low = fuzz.trapmf(variable_range,
#                               [0, 0, a, b])
#             mid = fuzz.trimf(variable_range,
#                              [a, b, c])
#             high = fuzz.trapmf(variable_range,
#                                [b, c, 1, 1])
#             variables_fuzzy_sets[i] = (low, mid, high)
#
#         return variables_fuzzy_sets
#
#     def fit_wm(self, x, y, clu_paras):
#         rules = {}
#         fuzzy_sets = self.fuzzy_sets_definition(x, clu_paras)
#         variable_range = self.variable_range
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
#         self.rule_base = rules
#         self.fuzzy_sets = fuzzy_sets
#
#     def predict_all_rules(self, sample):
#         max_match_degree = 0
#         predicted_class = None
#         best_rule = None
#
#         rules = self.rule_base
#         fuzzy_sets = self.fuzzy_sets
#         variable_range = self.variable_range
#         labels = set(rules.values())
#
#         for current_class in labels:
#             current_class_match_degree = 0
#             current_max_rule_match_degree = -np.inf
#             for rule_key, rule_class in rules.items():
#                 if rule_class == current_class:
#                     match_degree = 1
#                     for i, feature_value in enumerate(sample):
#                         if rule_key[i] == "Low":
#                             membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value)
#                         elif rule_key[i] == "Mid":
#                             membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value)
#                         elif rule_key[i] == "High":
#                             membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value)
#                         match_degree *= membership
#                     if match_degree >= current_max_rule_match_degree:
#                         current_best_rule = rule_key, rule_class
#                         current_max_rule_match_degree = match_degree
#
#                     current_class_match_degree += match_degree
#
#             if current_class_match_degree >= max_match_degree:
#                 max_match_degree = current_class_match_degree
#                 predicted_class = current_class
#                 best_rule = current_best_rule
#                 max_rule_match_degree = current_max_rule_match_degree
#
#         return predicted_class, best_rule, max_rule_match_degree
#
#
#     def predict_all_rules_mbcw(self, sample, causal_weight):
#         max_match_degree = 0
#         predicted_class = None
#         best_rule = None
#
#         rules = self.rule_base
#         fuzzy_sets = self.fuzzy_sets
#         variable_range = self.variable_range
#         labels = set(rules.values())
#         fs_dict = defaultdict(float)
#
#         for current_class in labels:
#             current_class_match_degree = 0
#             current_max_rule_match_degree = -np.inf
#             for rule_key, rule_class in rules.items():
#                 if rule_class == current_class:
#                     match_degree = 1
#                     for i, feature_value in enumerate(sample):
#                         if rule_key[i] == "Low":
#                             membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][0], feature_value)
#                         elif rule_key[i] == "Mid":
#                             membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][1], feature_value)
#                         elif rule_key[i] == "High":
#                             membership = fuzz.interp_membership(variable_range, fuzzy_sets[i][2], feature_value)
#                         match_degree *= membership
#                     if match_degree >= current_max_rule_match_degree:
#                         current_best_rule = rule_key, rule_class
#                         current_max_rule_match_degree = match_degree
#
#                     current_class_match_degree += match_degree
#             fs_dict[current_class] = current_class_match_degree*causal_weight
#
#             # if current_class_match_degree >= max_match_degree:
#             #     max_match_degree = current_class_match_degree
#             #     predicted_class = current_class
#             #     best_rule = current_best_rule
#             #     max_rule_match_degree = current_max_rule_match_degree
#
#         return fs_dict
#
#
#     def predict(self, sample):
#         max_match_degree = -np.inf
#         predicted_class = None
#         best_rule = None
#
#         rules = self.rule_base
#         fuzzy_sets = self.fuzzy_sets
#         variable_range = self.variable_range
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