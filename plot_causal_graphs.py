from mablar import *

from mablar import *

if __name__ == '__main__':
    data_set_name = 'sachs'
    data_set_path = 'Datasets/' + data_set_name + '.csv'
    cg_save_path = data_set_name + '_test.png'

    # variable names list: ['raf', 'mek', 'plc', 'pip2', 'pip3', 'erk', 'akt', 'pka', 'pkc', 'p38', 'jnk']
    target_variable = 2  # the index of variables, starts from 0
    test_sample_index = 1  # the index of the test sample

    CD = CausalDiscovery()
    RG = RuleGeneration()
    cg_model = lingam.ICALiNGAM(random_state=7)

    gt_dag, node_names_list = CD.get_gt_sachs_causal_graph(data_set_path)
    print('The target vairable is: ', node_names_list[target_variable])

    causal_matrix = CD.cg_matrix_to_adjacent_matrix(gt_dag.graph)
    gt_causal_matrix = causal_matrix.astype(np.float32)
    num_variables = gt_causal_matrix.shape[0]
    # weighted_causal_matrix = CD.creat_random_weighted_causal_matrix(ground_truth_causal_matrix)
    print(node_names_list)

    original_data = pd.read_csv(data_set_path)
    scalar = MinMaxScaler()
    normalise_data = scalar.fit_transform(original_data)
    cla_normalised_data = create_classification_data_from_regression_data(normalise_data, target_variable)

    cg_model.fit(cla_normalised_data)
    weighted_causal_matrix = cg_model.adjacency_matrix_
    weighted_causal_matrix = weighted_causal_matrix.T
    predicted_dag = CD.create_dag_from_matrix(weighted_causal_matrix, node_names_list)
    CD.show_causal_graph(predicted_dag, cg_save_path)

    all_causal_paths = CD.weighted_causal_paths_identification(weighted_causal_matrix, target_variable)


    all_causal_weights = CD.calculate_all_causal_path_weights(all_causal_paths, weighted_causal_matrix)

    original_y = cla_normalised_data[:, target_variable]

    variable_range = np.linspace(0, 1, 1000)

    test_sample_x = cla_normalised_data[test_sample_index, :]
    test_sample_y = cla_normalised_data[test_sample_index, target_variable]
    print('the ground truth label is: ', test_sample_y)

    node_names_list_str = ['raf', 'mek', 'plc', 'pip2', 'pip3', 'erk', 'akt', 'pka', 'pkc', 'p38', 'jnk']

    # the wm algorithm
    wm_variables = list(range(num_variables))
    wm_variables.remove(target_variable)
    wm_causal_matrix = CD.mb_to_causal_matrix_target(wm_variables, num_variables, target_variable)
    wm_dag = CD.create_dag_from_matrix(wm_causal_matrix, node_names_list)
    CD.show_causal_graph(wm_dag, save_path='wm_sachs_pip3.png')
    wm_causal_accuracy, wm_causal_precision, wm_causal_recall, wm_causal_f1 = CD.evaluate_causal_matrix(
        gt_causal_matrix, wm_causal_matrix)
    print(wm_causal_accuracy, wm_causal_precision, wm_causal_recall, wm_causal_f1)

    # the mablar algorithm
    final_mb = find_markov_blanket(weighted_causal_matrix, target_variable)
    mb_causal_matrix = CD.mb_to_causal_matrix_target(final_mb, num_variables, target_variable)
    mb_dag = CD.create_dag_from_matrix(mb_causal_matrix, node_names_list)
    CD.show_causal_graph(mb_dag, save_path='mb_sachs_pip3.png')
    mb_causal_accuracy, mb_causal_precision, mb_causal_recall, mb_causal_f1 = CD.evaluate_causal_matrix(
        gt_causal_matrix, mb_causal_matrix)
    print(mb_causal_accuracy, mb_causal_precision, mb_causal_recall, mb_causal_f1)

    # the mablar_cd algorithm
    final_mbcd = find_direct_cause(weighted_causal_matrix, target_variable)
    mbcd_causal_matrix = CD.mb_to_causal_matrix_target(final_mbcd, num_variables, target_variable)
    mbcd_dag = CD.create_dag_from_matrix(mbcd_causal_matrix, node_names_list)
    CD.show_causal_graph(mbcd_dag, save_path='mbcd_sachs_pip3.png')

    mbcd_causal_accuracy, mbcd_causal_precision, mbcd_causal_recall, mbcd_causal_f1 = CD.evaluate_causal_matrix(
        gt_causal_matrix, mbcd_causal_matrix)
    print(mbcd_causal_accuracy, mbcd_causal_precision, mbcd_causal_recall, mbcd_causal_f1)

