from mablar import *


if __name__ == '__main__':
    data_set_name = 'sachs'
    data_set_path = 'Datasets/' + data_set_name + '.csv'
    cg_save_path = 'test_'+data_set_name+'.png'

    CD = CausalDiscovery()
    RG = RuleGeneration()

    dag, node_names_list = CD.get_gt_sachs_causal_graph(data_set_path)
    target_variable = 8  # the pkc variable, starts from 0

    causal_matrix = CD.cg_matrix_to_adjacent_matrix(dag.graph)
    causal_matrix = causal_matrix.astype(np.float32)
    weighted_causal_matrix = CD.creat_random_weighted_causal_matrix(causal_matrix)
    # weighted_causal_matrix = copy.deepcopy(causal_matrix)

    # print(weighted_causal_matrix)
    print(node_names_list)

    original_data = pd.read_csv(data_set_path)
    scalar = MinMaxScaler()
    normalise_data = scalar.fit_transform(original_data)
    cla_normalised_data = create_classification_data_from_regression_data(normalise_data, target_variable)

    cla_node_names_list = create_moved_node_names_list(node_names_list, target_variable)
    all_causal_paths = CD.weighted_causal_paths_identification(weighted_causal_matrix, target_variable)
    print(all_causal_paths)

    all_weights = CD.calculate_all_causal_path_weights(all_causal_paths, weighted_causal_matrix)

    original_y = cla_normalised_data[:, target_variable]

    variable_range = np.linspace(0, 1, 1000)

    # all_fuzzy_system, all_fuzzy_set = RG.mablar_cw_wm_train(cla_normalised_data, original_y, all_causal_paths, weighted_causal_matrix)
    # mbcw_accuracy_score = RG.mable_wm_wm_cross_validation(cla_normalised_data, original_y, all_causal_paths, all_weights, weighted_causal_matrix)
    # print('mbcw_accuracy_score: ', mbcw_accuracy_score)


    final_mb = find_markov_blanket(weighted_causal_matrix, target_variable)
    normalise_mb_x = copy.deepcopy(cla_normalised_data)
    normalise_mb_x = normalise_mb_x[:, final_mb]
    mb_accuracy_score = RG.wm_cross_validation(normalise_mb_x, original_y)
    print('mb_accuracy_score: ', mb_accuracy_score)

    # final_mbcd = find_direct_cause(weighted_causal_matrix, target_variable)
    # normalise_mbcd_x = copy.deepcopy(cla_normalised_data[:, final_mbcd])
    # mbcd_accuracy_score = RG.wm_cross_validation(normalise_mbcd_x, original_y)
    # print('mbcd_accuracy_score: ', mbcd_accuracy_score)
