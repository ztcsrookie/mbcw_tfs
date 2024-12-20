import copy

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from fuzzy_system import *
from mablar import *
from validation_functions import *


class paras():
    def __init__(self, clus_num=3, clu_init='random', clu_alg='lloyd', random_seed=3):  # lloyd elkan
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


def mablar_cw_wm_predict_vote(sample, all_fuzzy_systems, all_causal_path, causal_path_weights):
    num_fuzzy_systems = len(all_causal_path)
    all_predicts = []
    for i in range(num_fuzzy_systems):
        current_variable_set = all_causal_path[i]
        current_variable_set = np.array(current_variable_set)
        current_input = sample[current_variable_set]
        current_fuzzy_system = all_fuzzy_systems[i]
        current_pre, current_best_rule, current_match_degree = current_fuzzy_system.predict(current_input)
        all_predicts.append(current_pre)
    final_predict = weighted_voting(all_predicts, causal_path_weights)
    return final_predict


def mablar_cw_wm_predict_product(sample, all_fuzzy_systems, all_causal_path, causal_path_weights):
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


def multi_mbcw_cross_validation(normalised_x, original_y, paras, all_causal_paths, add_causal_weights,
                                weighted_add_causal_weights, product_causal_weights):
    '''
    Test the obtained rules using cross_validation
    :param normalised_data: a numpy matrix
    :return:
    '''

    random_seed = paras.random_seed

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    # skf = StratifiedKFold(n_splits=5)

    add_vote_accuracies = []
    add_vote_fold_results = {}

    weighted_add_vote_accuracies = []
    weighted_add_vote_fold_results = {}

    add_product_accuracies = []
    add_product_fold_results = {}

    weighted_add_product_accuracies = []
    weighted_add_product_fold_results = {}

    product_vote_accuracies = []

    product_product_accuracies = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(normalised_x)):
        # for fold_idx, (train_index, test_index) in enumerate(skf.split(normalised_x, original_y)):
        # 划分训练集和测试集
        X_train, X_test = normalised_x[train_index], normalised_x[test_index]
        y_train, y_test = original_y[train_index], original_y[test_index]

        fold_name = f'fold_{fold_idx + 1}'
        add_vote_fold_results[fold_name] = {}
        weighted_add_vote_fold_results[fold_name] = {}
        add_product_fold_results[fold_name] = {}
        weighted_add_product_fold_results[fold_name] = {}

        # 训练多个模糊系统
        all_fuzzy_systems = mablar_cw_wm_train(X_train, y_train, paras, all_causal_paths)

        # 使用规则对测试数据进行分类
        add_vote_correct_counts = 0
        weighted_add_vote_correct_counts = 0
        add_product_correct_counts = 0
        weighted_add_product_correct_counts = 0
        product_vote_correct_counts = 0
        product_product_correct_counts = 0

        num_test_samples = X_test.shape[0]
        for i in range(num_test_samples):
            sample = X_test[i, :]
            add_vote_prediction = mablar_cw_wm_predict_vote(sample, all_fuzzy_systems, all_causal_paths,
                                                            add_causal_weights)
            weighted_add_vote_prediction = mablar_cw_wm_predict_vote(sample, all_fuzzy_systems, all_causal_paths,
                                                                     weighted_add_causal_weights)
            product_vote_prediction = mablar_cw_wm_predict_vote(sample, all_fuzzy_systems, all_causal_paths,
                                                                product_causal_weights)
            add_product_prediction = mablar_cw_wm_predict_product(sample, all_fuzzy_systems, all_causal_paths,
                                                                  add_causal_weights)
            weighted_add_product_prediction = mablar_cw_wm_predict_product(sample, all_fuzzy_systems, all_causal_paths,
                                                                           weighted_add_causal_weights)
            product_product_prediction = mablar_cw_wm_predict_product(sample, all_fuzzy_systems, all_causal_paths,
                                                                      product_causal_weights)
            if add_vote_prediction == y_test[i]:
                add_vote_correct_counts += 1
            if weighted_add_vote_prediction == y_test[i]:
                weighted_add_vote_correct_counts += 1
            if add_product_prediction == y_test[i]:
                add_product_correct_counts += 1
            if weighted_add_product_prediction == y_test[i]:
                weighted_add_product_correct_counts += 1
            if product_vote_prediction == y_test[i]:
                product_vote_correct_counts += 1
            if product_product_prediction == y_test[i]:
                product_product_correct_counts += 1

        # 计算并存储每次的准确率
        add_vote_accuracy = add_vote_correct_counts / num_test_samples
        add_vote_accuracies.append(add_vote_accuracy)

        weighted_add_vote_accuracy = weighted_add_vote_correct_counts / num_test_samples
        weighted_add_vote_accuracies.append(weighted_add_vote_accuracy)

        product_vote_accuracy = product_vote_correct_counts / num_test_samples
        product_vote_accuracies.append(product_vote_accuracy)

        add_product_accuracy = add_product_correct_counts / num_test_samples
        add_product_accuracies.append(add_product_accuracy)

        weighted_add_product_accuracy = weighted_add_product_correct_counts / num_test_samples
        weighted_add_product_accuracies.append(weighted_add_product_accuracy)

        product_product_accuracy = product_product_correct_counts / num_test_samples
        product_product_accuracies.append(product_product_accuracy)

        # print("Fold accuracy:", accuracy)
        # fold_results[fold_name]['model'] = all_fuzzy_systems
        # fold_results[fold_name]['accuracy'] = vote_accuracy
    # 计算平均准确率
    add_vote_average_accuracy = np.mean(add_vote_accuracies)
    weighted_add_vote_average_accuracy = np.mean(weighted_add_vote_accuracies)
    product_vote_average_accuracy = np.mean(product_vote_accuracies)
    add_product_average_accuracy = np.mean(add_product_accuracies)
    weighted_add_product_average_accuracy = np.mean(weighted_add_product_accuracies)
    product_product_average_accuracy = np.mean(product_product_accuracies)

    # print("Current average accuracy is:", average_accuracy)
    return add_vote_average_accuracy, weighted_add_vote_average_accuracy, product_vote_average_accuracy, \
           add_product_average_accuracy, weighted_add_product_average_accuracy, product_product_average_accuracy


if __name__ == '__main__':
    # data_set_names = ['breast', 'ecoli', 'glass', 'iris', 'mammographic', 'pima_diabetes', 'wine', 'sachs_pip3', 'HTRU2']
    # data_set_names = ['breast', 'iris', 'mammographic', 'pima_diabetes']
    data_set_names = ['sachs_pkc']

    num_data_sets = len(data_set_names)
    num_frameworks = 6
    performance_results = np.zeros((num_data_sets, num_frameworks))
    # for data_set_name in data_set_names:
    for data_set_index in range(num_data_sets):
        data_set_name = data_set_names[data_set_index]
        print('Current data set is: ', data_set_name)

        best_add_vote_accuracy = 0
        best_weighted_add_vote_accuracy = 0
        best_product_vote_accuracy = 0
        best_add_product_accuracy = 0
        best_weighted_add_product_accuracy = 0
        best_product_product_accuracy = 0


        data_set_path = 'Datasets/' + data_set_name + '.csv'
        model_save_path = 'Results/Models/' + data_set_name + '.pkl'
        cg_save_path = 'Results/CGs/mbcw_best_cgs/' + data_set_name + '_Dibk.png'
        best_cg_save_path = 'Results/CGs/mbcw_best_cgs/' + data_set_name + '_bk_best.png'
        # performance_save_path = 'Results/Models/' + data_set_name + '.pkl'
        le_data, normalised_x, encoded_y, original_y = load_CD_data(data_set_path)
        for current_random_seed in range(5):
            for cd_measure in ['kernal', 'pwling']:
                try:
                    CD = CausalDiscovery()
                    # cd_model = lingam.ICALiNGAM(random_state=current_random_seed, max_iter=10000)
                    cd_model = lingam.DirectLiNGAM(random_state=current_random_seed, measure=cd_measure)
                    # cd_model = lingam.RCD()
                    cd_model.fit(le_data)
                    weighted_causal_matrix = copy.deepcopy(cd_model.adjacency_matrix_)
                    weighted_causal_matrix = weighted_causal_matrix.T
                    node_name_list = CD.get_node_names_list(data_set_path)
                    cd_dag = CD.create_dag_from_matrix(weighted_causal_matrix, node_name_list)
                    CD.show_causal_graph(cd_dag, cg_save_path)
                except:
                    print("CD wrong on:", data_set_name)
                    continue

                # Construct MBCW subset
                all_causal_paths = CD.weighted_causal_paths_identification(weighted_causal_matrix, -1)
                if not all_causal_paths:
                    print(data_set_name, 'has no direct causes.')
                    continue
                all_complete_causal_path = copy.deepcopy(all_causal_paths)
                product_all_weights = CD.calculate_all_causal_path_weights_product(all_complete_causal_path,
                                                                                   weighted_causal_matrix)
                add_all_weights = CD.calculate_all_causal_path_weights_add(all_complete_causal_path, weighted_causal_matrix)
                weighted_add_all_weights = CD.calculate_all_causal_path_weights_weighted(all_complete_causal_path,
                                                                                         weighted_causal_matrix)

                # Find the causal graph of the given data set
                for current_clu_init in ['k-means++', 'random']:
                    for current_clu_alg in ['lloyd', 'elkan']:
                        # print('Current paras: ', current_random_seed, current_clu_init, current_clu_alg, cd_measure)
                        current_paras = paras(clu_init=current_clu_init, clu_alg=current_clu_alg,
                                              random_seed=current_random_seed)
                        try:
                            add_vote_scores, weighted_add_vote_scores, product_vote_scores, \
                            add_product_scores, weighted_add_product_scores, product_product_scores \
                                = multi_mbcw_cross_validation(normalised_x, original_y, current_paras, all_causal_paths,
                                                              add_all_weights, weighted_add_all_weights,
                                                              product_all_weights)
                            if add_vote_scores >= best_add_vote_accuracy:
                                best_add_vote_accuracy = add_vote_scores
                            if weighted_add_vote_scores >= best_weighted_add_vote_accuracy:
                                best_weighted_add_vote_accuracy = weighted_add_vote_scores
                            if product_vote_scores >= best_product_vote_accuracy:
                                best_product_vote_accuracy = product_vote_scores
                            if add_product_scores >= best_add_product_accuracy:
                                best_add_product_accuracy = add_product_scores
                            if weighted_add_product_scores >= best_weighted_add_product_accuracy:
                                best_weighted_add_product_accuracy = weighted_add_product_scores

                                best_cg = cd_dag
                                CD.show_causal_graph(best_cg, best_cg_save_path)

                            if product_product_scores >= best_product_product_accuracy:
                                best_product_product_accuracy = product_product_scores
                        except:
                            print('Something wrong on: ', data_set_name)
    print('best of AV', data_set_name, ' is: ', best_add_vote_accuracy)
    print('best of WAV', data_set_name, ' is: ', best_weighted_add_vote_accuracy)
    print('best of PV', data_set_name, ' is: ', best_product_vote_accuracy)
    print('best of AP', data_set_name, ' is: ', best_add_product_accuracy)
    print('best of WAP', data_set_name, ' is: ', best_weighted_add_product_accuracy)
    print('best of PP', data_set_name, ' is: ', best_product_product_accuracy)
    performance_results[data_set_index, 0] = best_add_vote_accuracy
    performance_results[data_set_index, 1] = best_weighted_add_vote_accuracy
    performance_results[data_set_index, 2] = best_product_vote_accuracy
    performance_results[data_set_index, 3] = best_add_product_accuracy
    performance_results[data_set_index, 4] = best_weighted_add_product_accuracy
    performance_results[data_set_index, 5] = best_product_product_accuracy
    np.savetxt('Results/Performance/04_12_multi_cws_sachs_pkc_di.csv', performance_results, delimiter=',', fmt='%.4f')