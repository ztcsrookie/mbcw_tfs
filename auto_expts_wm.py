from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from fuzzy_system import *
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
    print("Current average accuracy is:", average_accuracy)
    return fold_results, average_accuracy

if __name__ == '__main__':
    # data_set_names = ['authorship', 'pima_diabetes', 'mammographic', 'HTRU2', 'ecoli', 'wine']
    data_set_names = ['glass']
    # for data_set_name in data_set_names:
    for data_set_name in tqdm(data_set_names, desc="Data Set Progress"):
        best_accuracy = 0
        print('Current data set is: ', data_set_name)
        for current_clu_init in tqdm(['k-means++', 'random'], desc="init"):
            for current_clu_alg in tqdm(['lloyd', 'elkan'], desc="algorithms"):
                for current_random_seed in range(5):
                    data_set_path = 'Datasets/' + data_set_name + '.csv'
                    model_save_path = 'Results/Models/' + data_set_name + '.pkl'
                    # performance_save_path = 'Results/Models/' + data_set_name + '.pkl'
                    normalised_x, encoded_y, original_y = load_data(data_set_path)
                    current_paras = paras(clu_init=current_clu_init, clu_alg=current_clu_alg, random_seed=current_random_seed)
                    results, accuracy_score = wm_cross_validation(normalised_x, original_y, current_paras)
                    if accuracy_score > best_accuracy:
                        best_accuracy = accuracy_score
                        best_clu_init = current_clu_init
                        best_clu_alg = current_clu_alg
                        best_seed = current_random_seed

        print('best of', data_set_name, ' is: ', best_accuracy)
        print('best_init is: ', best_clu_init)
        print('best_alg is: ', best_clu_alg)
        print('best_seed_is: ', best_seed)
