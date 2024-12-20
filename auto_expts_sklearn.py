import pandas as pd

from mablar import *
from validation_functions import *

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from tqdm import tqdm

if __name__ == '__main__':
    # 读取CSV数据
    # data_set_names = ['iris', 'breast', 'pima_diabetes', 'mammographic', 'wine', 'ecoli', 'HTRU2', 'page_blocks']
    data_set_names = ['wine']
    for data_set_name in tqdm(data_set_names, desc="Data Set Progress"):
        data_set_path = 'Datasets/' + data_set_name + '.csv'
        original_data = pd.read_csv(data_set_path)
        data = original_data.iloc[:, [2, 4]]
        label_column = data.columns[-1]
        # for data_set_name in data_set_names:
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
                # # Find the causal graph of the given data set
                #
                # bk_pima_matrix = np.full((9, 9), -1)
                # bk_pima_matrix[5, 7] = 1
                # bk_pima_matrix[4, 7] = 1
                # bk_pima_matrix[2, 7] = 1
                # bk_pima_matrix[1, 7] = 1
                # bk_pima_matrix[8, 2] = 1
                # bk_pima_matrix[8, 5] = 1
                # bk_pima_matrix[8, 6] = 1
                # bk_pima_matrix[8, 7] = 1

                CD = CausalDiscovery()
                # cd_model = lingam.ICALiNGAM(random_state=current_random_seed, max_iter=10000)
                cd_model = lingam.DirectLiNGAM(random_state=current_random_seed, measure='kernal')
                cd_model.fit(le_data)
                weighted_causal_matrix = copy.deepcopy(cd_model.adjacency_matrix_)
                weighted_causal_matrix = weighted_causal_matrix.T
                node_name_list = CD.get_node_names_list(data_set_path)
                cd_dag = CD.create_dag_from_matrix(weighted_causal_matrix, node_name_list)
                # Construct MB&MBCD subset
                final_mb = find_markov_blanket(weighted_causal_matrix)
                final_mbcd = find_direct_cause(weighted_causal_matrix)

                normalised_mb_x = copy.deepcopy(normalised_x[:, final_mb])
                normalised_mbcd_x = copy.deepcopy(normalised_x[:, final_mbcd])

                # svm_classifier = SVC(kernel='linear')
                # cv_scores = cross_val_score(svm_classifier, X, y, cv=5)

                tree_classifier = DecisionTreeClassifier()
                cv_scores = cross_val_score(tree_classifier, normalised_x, original_y, cv=5)
                print("Average Score of", data_set_name, 'is: ', cv_scores.mean())

                tree_mb_classifier = DecisionTreeClassifier()
                mb_cv_scores = cross_val_score(tree_mb_classifier, normalised_x, original_y, cv=5)
                print("MB Average Score of", data_set_name, 'is: ', mb_cv_scores.mean())

                tree_mbcd_classifier = DecisionTreeClassifier()
                mbcd_cv_scores = cross_val_score(tree_mbcd_classifier, normalised_x, original_y, cv=5)
                print("MBCD Average Score of", data_set_name, 'is: ', mbcd_cv_scores.mean())

                print(normalised_mb_x.shape)
                print(normalised_mbcd_x.shape)
                print(normalised_x.shape)


                    # 打印交叉验证得分

