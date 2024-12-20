import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from fuzzy_system import *
from mablar import *
from validation_functions import *


class paras():
    def __init__(self, clus_num=3, clu_init='random', clu_alg='lloyd', random_seed = 3): #lloyd elkan
        self.clus_num = clus_num
        self.clu_init = clu_init
        self.clu_alg = clu_alg
        self.random_seed = 3

def wm_cross_validation(normalised_x, original_y, pars):
    '''
    Test the obtained rules using cross_validation
    :param normalised_data: a numpy matrix
    :return:
    '''

    random_seed = pars.random_seed

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    # kf = KFold(n_splits=5)
    skf = StratifiedKFold(n_splits=5)
    accuracies = []
    fold_results = {}

    variable_range = np.linspace(0, 1, 10000)

    for fold_idx, (train_index, test_index) in enumerate(kf.split(normalised_x)):
        # print('Current fold is: ', fold_idx)
    # for fold_idx, (train_index, test_index) in enumerate(skf.split(normalised_x, original_y)):
        # 划分训练集和测试集
        X_train, X_test = normalised_x[train_index], normalised_x[test_index]
        y_train, y_test = original_y[train_index], original_y[test_index]

        current_fuzzy_system = FuzzySystem(variable_range=np.linspace(0, 1, 10000))

        fold_name = f'fold_{fold_idx + 1}'
        fold_results[fold_name] = {}

        # 使用Wang-Mendel算法从训练数据中提取规则
        current_fuzzy_system.fit_wm(X_train, y_train, pars)
        # predictions = []
        correct_counts = 0
        num_test_samples = X_test.shape[0]
        # 使用规则对测试数据进行分类
        for i in range(num_test_samples):
            sample = X_test[i, :]
            prediction, best_rule, max_match_degree = current_fuzzy_system.predict(sample)
            # prediction, best_rule, max_match_degree = current_fuzzy_system.predict_all_rules(sample)
            if prediction == y_test[i]:
                correct_counts += 1

        # 计算并存储每次的准确率
        accuracy = correct_counts/num_test_samples
        accuracies.append(accuracy)
        # print("Fold accuracy:", accuracy)
        fold_results[fold_name]['model'] = current_fuzzy_system
        fold_results[fold_name]['accuracy'] = accuracy

    # 计算平均准确率
    average_accuracy = np.mean(accuracies)
    # print("Current average accuracy is:", average_accuracy)
    return fold_results, average_accuracy


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


def mablar_cw_wm_predict(sample, all_fuzzy_systems, all_causal_path, all_weights):
    num_fuzzy_systems = len(all_causal_path)
    all_predicts = []
    for i in range(num_fuzzy_systems):
        current_variable_set = all_causal_path[i]
        current_variable_set = np.array(current_variable_set)
        current_input = sample[current_variable_set]
        current_fuzzy_system = all_fuzzy_systems[i]
        current_pre, current_best_rule, current_match_degree = current_fuzzy_system.predict(current_input)
        all_predicts.append(current_pre)
    final_predict = weighted_voting(all_predicts, all_weights)
    return final_predict


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


def mbcw_cross_validation(normalised_x, original_y, paras, all_causal_paths, all_causal_weights):
    '''
    Test the obtained rules using cross_validation
    :param normalised_data: a numpy matrix
    :return:
    '''

    random_seed = paras.random_seed

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    skf = StratifiedKFold(n_splits=5)
    accuracies = []
    fold_results = {}

    variable_range = np.linspace(0, 1, 10000)

    for fold_idx, (train_index, test_index) in enumerate(kf.split(normalised_x)):
    # for fold_idx, (train_index, test_index) in enumerate(skf.split(normalised_x, original_y)):
        # 划分训练集和测试集
        X_train, X_test = normalised_x[train_index], normalised_x[test_index]
        y_train, y_test = original_y[train_index], original_y[test_index]

        fold_name = f'fold_{fold_idx + 1}'
        fold_results[fold_name] = {}
        #训练多个模糊系统
        all_fuzzy_systems = mablar_cw_wm_train(X_train, y_train, paras, all_causal_paths)

        # 使用规则对测试数据进行分类
        correct_counts = 0
        num_test_samples = X_test.shape[0]
        for i in range(num_test_samples):
            sample = X_test[i, :]
            prediction = mablar_cw_wm_predict(sample, all_fuzzy_systems, all_causal_paths, all_causal_weights)
            # prediction, best_rule, max_match_degree = current_fuzzy_system.predict_all_rules(sample)
            if prediction == y_test[i]:
                correct_counts += 1

        # 计算并存储每次的准确率
        accuracy = correct_counts/num_test_samples
        accuracies.append(accuracy)
        # print("Fold accuracy:", accuracy)
        fold_results[fold_name]['model'] = all_fuzzy_systems
        fold_results[fold_name]['accuracy'] = accuracy
    # 计算平均准确率
    average_accuracy = np.mean(accuracies)
    # print("Current average accuracy is:", average_accuracy)
    return fold_results, average_accuracy

if __name__ == '__main__':
    # data_set_names = ['breast', 'ecoli', 'glass', 'iris', 'mammographic', 'pima_diabetes', 'wine', 'sachs_pip3', 'HTRU2']
    data_set_names = [ 'pima_diabetes']
    num_data_sets = len(data_set_names)
    num_frameworks = 3 #wm, mb, mbcd
    performance_results = np.zeros((num_data_sets, num_frameworks))
    # for data_set_name in data_set_names:
    for data_set_index in range(num_data_sets):
        data_set_name = data_set_names[data_set_index]
        print('Current data set is: ', data_set_name)

        best_wm_accuracy = 0
        best_mb_accuracy = 0
        best_mbcd_accuracy = 0
        best_mbcw_accuracy = 0

        data_set_path = 'Datasets/' + data_set_name + '.csv'
        model_save_path = 'Results/Models/' + data_set_name + '.pkl'
        cg_save_path = 'Results/CGs/mb_best_cgs/' + data_set_name + '_04_12_di.png'
        # performance_save_path = 'Results/Models/' + data_set_name + '.pkl'
        le_data, normalised_x, encoded_y, original_y = load_CD_data(data_set_path)

        for current_random_seed in range(5):
            for measure_way in ['kernal']:
                #Find the causal graph of the given data set

                bk_pima_matrix = np.full((9, 9), -1)
                bk_pima_matrix[5, 7] = 1
                bk_pima_matrix[4, 7] = 1
                bk_pima_matrix[2, 7] = 1
                bk_pima_matrix[1, 7] = 1
                bk_pima_matrix[8, 2] = 1
                bk_pima_matrix[8, 5] = 1
                bk_pima_matrix[8, 6] = 1
                bk_pima_matrix[8, 7] = 1

                CD = CausalDiscovery()
                # cd_model = lingam.ICALiNGAM(random_state=current_random_seed, max_iter=10000)
                cd_model = lingam.DirectLiNGAM(random_state=current_random_seed, prior_knowledge=bk_pima_matrix,
                                               apply_prior_knowledge_softly=True, measure='kernal')
                cd_model.fit(le_data)
                weighted_causal_matrix = copy.deepcopy(cd_model.adjacency_matrix_)
                weighted_causal_matrix = weighted_causal_matrix.T
                node_name_list = CD.get_node_names_list(data_set_path)
                cd_dag = CD.create_dag_from_matrix(weighted_causal_matrix, node_name_list)
                #Construct MB&MBCD subset
                final_mb = find_markov_blanket(weighted_causal_matrix)
                final_mbcd = find_direct_cause(weighted_causal_matrix)

                normalised_mb_x = copy.deepcopy(normalised_x[:, final_mb])
                normalised_mbcd_x = copy.deepcopy(normalised_x[:, final_mbcd])

                # Construct MBCW subset
                all_causal_paths = CD.weighted_causal_paths_identification(weighted_causal_matrix, -1)
                if not all_causal_paths:
                    print(data_set_name, 'has no direct causes.')
                    continue
                # all_weights = CD.calculate_all_causal_path_weights_product(all_causal_paths, weighted_causal_matrix)
                # all_weights = CD.calculate_all_causal_path_weights_add(all_causal_paths, weighted_causal_matrix)
                # all_weights = CD.calculate_all_causal_path_weights_weighted(all_causal_paths, weighted_causal_matrix)
                for current_clu_init in ['k-means++', 'random']:
                    for current_clu_alg in ['lloyd', 'elkan']:
                        # print('Current paras: ', current_random_seed, current_clu_init, current_clu_alg)
                        current_paras = paras(clu_init=current_clu_init, clu_alg=current_clu_alg,
                                              random_seed=current_random_seed)
                        try:
                            wm_results, wm_accuracy_score = wm_cross_validation(normalised_x, original_y, current_paras)
                            if wm_accuracy_score > best_wm_accuracy:
                                best_wm_accuracy = wm_accuracy_score
                        except:
                            print('WM, ', data_set_name)

                        try:
                            mb_results, mb_accuracy_score = wm_cross_validation(normalised_mb_x, original_y, current_paras)
                            if mb_accuracy_score > best_mb_accuracy:
                                best_mb_accuracy = mb_accuracy_score
                                best_dag = cd_dag
                        except:
                            print('MB, ', data_set_name)

                        try:
                            if not normalised_mbcd_x.shape[1]:
                                print('No direct causes in ', data_set_name)
                            else:
                                mbcd_results, mbcd_accuracy_score = wm_cross_validation(normalised_mbcd_x, original_y, current_paras)
                                if mbcd_accuracy_score > best_mbcd_accuracy:
                                    best_mbcd_accuracy = mbcd_accuracy_score
                        except:
                            print('MBCD, ', data_set_name)

                        # try:
                        #     mbcw_results, mbcw_accuracy_score = mbcw_cross_validation(normalised_x, original_y,
                        #                                                               current_paras, all_causal_paths,
                        #                                                               all_weights)
                        #     if mbcw_accuracy_score >= best_mbcw_accuracy:
                        #         best_mbcw_accuracy = mbcw_accuracy_score
                        #         best_dag = cd_dag
                        #         best_clu_init = current_clu_init
                        #         best_clu_alg = current_clu_alg
                        #         best_seed = current_random_seed
                        # except:
                        #     print('MBCW, ', data_set_name)

        CD.show_causal_graph(best_dag, cg_save_path)
        print('WM best of', data_set_name, ' is: ', best_wm_accuracy)
        print('MB best of', data_set_name, ' is: ', best_mb_accuracy)
        print('MBCD best of', data_set_name, ' is: ', best_mbcd_accuracy)
        # print('MBCW best of', data_set_name, ' is: ', best_mbcw_accuracy)
        performance_results[data_set_index, 0] = best_wm_accuracy
        performance_results[data_set_index, 1] = best_mb_accuracy
        performance_results[data_set_index, 2] = best_mbcd_accuracy
        # performance_results[data_set_index, 3] = best_mbcw_accuracy
    np.savetxt('Results/Performance/04_12_mablars_pima.csv', performance_results, delimiter=',', fmt='%.4f')