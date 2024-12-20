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

        final_mb = find_markov_blanket(weighted_causal_matrix)
        final_mbcd = find_direct_cause(weighted_causal_matrix)
        normalised_mb_x = copy.deepcopy(normalised_x[:, final_mb])
        normalised_mbcd_x = copy.deepcopy(normalised_x[:, final_mbcd])

    #Find the causal graph of the given data set
    for current_clu_init in ['k-means++']: #, 'random'
        for current_clu_alg in ['lloyd']: # , 'elkan'
            current_paras = paras(clu_init=current_clu_init, clu_alg=current_clu_alg,
                                  random_seed=4)
            current_wm = FuzzySystem(variable_range=np.linspace(0, 1, 10000))
            current_wm.fit_wm(normalised_x, original_y, current_paras)


            current_mb = FuzzySystem(variable_range=np.linspace(0, 1, 10000))
            current_mb.fit_wm(normalised_mb_x, original_y, current_paras)

            current_mbcd = FuzzySystem(variable_range=np.linspace(0, 1, 10000))
            current_mbcd.fit_wm(normalised_mbcd_x, original_y, current_paras)

            # sample_index = random.randint(1, 700)
            sample_index = 30
            sample1 = normalised_x[sample_index, :]
            mb_sample1 = normalised_x[sample_index, final_mb]
            mbcd_sample1 = normalised_x[sample_index, final_mbcd]
            label1 = original_y[sample_index]

            predicted_class_wm, best_rule_wm, max_match_degree_wm = current_wm.predict(sample1)
            print(predicted_class_wm, best_rule_wm, max_match_degree_wm)

            predicted_class_mb, best_rule_mb, max_match_degree_mb = current_mb.predict(mb_sample1)
            print(predicted_class_mb, best_rule_mb, max_match_degree_mb)

            predicted_class_mbcd, best_rule_mbcd, max_match_degree_mbcd = current_mb.predict(mbcd_sample1)
            print(predicted_class_mbcd, best_rule_mbcd, max_match_degree_mbcd)