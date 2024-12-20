import copy

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from fuzzy_system import *
from mablar import *
from validation_functions import *
import random


class paras():
    def __init__(self, clus_num=3, clu_init='random', clu_alg='lloyd', random_seed = 3): #lloyd elkan
        self.clus_num = clus_num
        self.clu_init = clu_init
        self.clu_alg = clu_alg
        self.random_seed = 3

def mablar_cw_wm_train(normalised_x, y, paras, all_causal_paths):
    '''
    :param original_x:the normalised original x, i.e., unordered and including the target variable data set.
    :param y:
    :param all_causal_paths:
    :param weighted_causal_matrix:
    :param num_mf:
    :param variable_range:
    :return:
    '''
    num_fuzzy_systems = len(all_causal_paths)
    all_fuzzy_systems = []
    for i in range(num_fuzzy_systems):
        variable_set = all_causal_paths[i]
        current_fuzzy_system = FuzzySystem()
        train_x = normalised_x[:, variable_set]
        current_fuzzy_system.fit_wm(train_x, y, paras)
        all_fuzzy_systems.append(current_fuzzy_system)
    return all_fuzzy_systems


def weighted_voting(classifiers_output, all_weights):
    if len(classifiers_output) != len(all_weights):
        raise ValueError("The number of classifiers' outputs and weights should be the same")

    # 初始化得票数为0
    votes = defaultdict(float)

    # 为每个类别计算得票数
    for cls_output, weight in zip(classifiers_output, all_weights):
        votes[cls_output] += weight

    # 返回得票数最多的类别
    return max(votes, key=votes.get)

def mablar_cw_wm_predict_vote_local(sample, all_fuzzy_systems, all_causal_path, causal_path_weights):
    num_fuzzy_systems = len(all_causal_path)
    all_predicts = []
    max_match_degree = float('-inf')
    for i in range(num_fuzzy_systems):
        current_variable_set = all_causal_path[i]
        current_weight = causal_path_weights[i]
        current_variable_set = np.array(current_variable_set)
        current_input = sample[current_variable_set]
        current_fuzzy_system = all_fuzzy_systems[i]
        current_pre, current_best_rule, current_match_degree = current_fuzzy_system.predict(current_input)
        all_predicts.append(current_pre)
        current_weights_max_degree = current_match_degree*current_weight
        if current_weights_max_degree > max_match_degree:
            max_match_degree = current_weights_max_degree
            max_degree_rule = current_best_rule
            max_variable_set = current_variable_set
            # print('Degree:', current_weights_max_degree)
            # print('Variable:', current_variable_set)
            # print('Rule:', max_degree_rule)
    final_predict = weighted_voting(all_predicts, causal_path_weights)
    # print(final_predict)
    return final_predict, max_match_degree, max_degree_rule, max_variable_set


def mablar_cw_wm_predict_product_local(sample, all_fuzzy_systems, all_causal_path, causal_path_weights):
    num_fuzzy_systems = len(all_causal_path)
    all_fs_dicts = []
    for i in range(num_fuzzy_systems):
        current_weight = causal_path_weights[i]
        current_variable_set = all_causal_path[i]
        current_variable_set = np.array(current_variable_set)
        current_input = sample[current_variable_set]
        current_fuzzy_system = all_fuzzy_systems[i]
        current_weighted_fs_dict = current_fuzzy_system.predict_all_rules_mbcw(current_input, current_weight)
        all_fs_dicts.append(current_weighted_fs_dict)

    sum_fs_dicts = defaultdict(float)
    for each_fs_dic in all_fs_dicts:
        for key, value in each_fs_dic.items():
            sum_fs_dicts[key] += value
    final_predict = max(sum_fs_dicts, key=sum_fs_dicts.get)

    # product_fs_dict = {}
    # for key, value in all_fs_dicts[0].items():
    #     product_fs_dict[key] = value
    # for each_fs_dic in all_fs_dicts:
    #     for key, value in each_fs_dic.items():
    #         product_fs_dict[key] *= value
    # final_predict = max(product_fs_dict, key=sum_fs_dicts.get)

    return final_predict

if __name__ == '__main__':
    # data_set_names = ['breast', 'ecoli', 'glass', 'iris', 'mammographic', 'pima_diabetes', 'wine', 'sachs_pip3', 'HTRU2']
    data_set_names = ['mammographic']

    num_data_sets = len(data_set_names)
    num_frameworks = 9
    # for data_set_name in data_set_names:
    for data_set_index in range(num_data_sets):
        data_set_name = data_set_names[data_set_index]
        print('Current data set is: ', data_set_name)

        data_set_path = 'Datasets/' + data_set_name + '.csv'
        cg_save_path = 'Results/CGs/' + data_set_name + '_DiLi_explanations.png'
        # performance_save_path = 'Results/Models/' + data_set_name + '.pkl'
        le_data, normalised_x, encoded_y, original_y = load_CD_data(data_set_path)

        CD = CausalDiscovery()
        node_name_list = CD.get_node_names_list(data_set_path)
        # cd_model = lingam.ICALiNGAM(random_state=current_random_seed, max_iter=10000)
        cd_model = lingam.DirectLiNGAM(random_state=3, measure='pwling')
        cd_model.fit(le_data)

        cg_dot = make_dot(cd_model.adjacency_matrix_, labels=node_name_list)
        cg_dot.render('test_graph1', format='png', cleanup=True)

        weighted_causal_matrix = copy.deepcopy(cd_model.adjacency_matrix_)
        weighted_causal_matrix = weighted_causal_matrix.T

        cd_dag = CD.create_dag_from_matrix(weighted_causal_matrix, node_name_list)
        CD.show_causal_graph(cd_dag, cg_save_path)

    # Construct MBCW subset
    all_causal_paths = CD.weighted_causal_paths_identification(weighted_causal_matrix, -1)
    all_complete_causal_path = copy.deepcopy(all_causal_paths)
    #Calculate weights
    product_all_weights = CD.calculate_all_causal_path_weights_product(all_complete_causal_path, weighted_causal_matrix)
    add_all_weights = CD.calculate_all_causal_path_weights_add(all_complete_causal_path, weighted_causal_matrix)
    weighted_add_all_weights = CD.calculate_all_causal_path_weights_weighted(all_complete_causal_path,
                                                                             weighted_causal_matrix)

    #Find the causal graph of the given data set
    for current_clu_init in ['k-means++']: #, 'random'
        for current_clu_alg in ['lloyd']: # , 'elkan'
            current_paras = paras(clu_init=current_clu_init, clu_alg=current_clu_alg,
                                  random_seed=4)
            all_fuzzy_systems = mablar_cw_wm_train(normalised_x, original_y, current_paras, all_causal_paths)
            # for sample_index in range(600):
            # sample_index = random.randint(1, 700)
            # print(sample_index)
            sample_index = 30
            sample1 = normalised_x[sample_index, :]
            label1 = original_y[sample_index]
            print('Sample is', sample1, label1)

            # print('Add')
            # final_predict1, max_match_degree1, max_degree_rule1, max_variable_set1 = mablar_cw_wm_predict_vote_local(
            #     sample1, all_fuzzy_systems, all_causal_paths, add_all_weights)
            # for i in max_variable_set1:
            #     print(node_name_list[i], end=' ')
            # print('Rule: ', max_degree_rule1, end=' ')
            # print(max_match_degree1)

            print('WA')
            final_predict2, max_match_degree2, max_degree_rule2, max_variable_set2 = mablar_cw_wm_predict_vote_local(
                sample1, all_fuzzy_systems, all_causal_paths, weighted_add_all_weights)
            # if len(max_variable_set2) != 3:
            #     print(sample_index)
            for i in max_variable_set2:
                print(node_name_list[i], end=' ')
            print('Rule: ', max_degree_rule2)
            print(max_match_degree2, end=' ')
            #
            # print('Product')
            # final_predict3, max_match_degree3, max_degree_rule3, max_variable_set3 = mablar_cw_wm_predict_vote_local(
            #     sample1, all_fuzzy_systems, all_causal_paths, product_all_weights)
            # for i in max_variable_set3:
            #     print(node_name_list[i], end=' ')
            # print('Rule: ', max_degree_rule3)
            # print(max_match_degree3, end=' ')

            # sample2 = normalised_x[300, :]
            # label2 = original_y[300]

