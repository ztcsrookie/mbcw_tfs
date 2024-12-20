import numpy as np
import skfuzzy as fuzz
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# 1. 加载iris数据集
iris = load_iris()
data = iris.data
target = iris.target

# 2. 为每个特征定义模糊集

def define_fuzzy_set(feature_range):
    low = fuzz.trimf(feature_range, [feature_range[0], feature_range[0], feature_range[int(2/3*len(feature_range))]])
    mid = fuzz.trimf(feature_range, [feature_range[0], feature_range[int(len(feature_range)/2)], feature_range[-1]])
    high = fuzz.trimf(feature_range, [feature_range[int(1/3*len(feature_range))], feature_range[-1], feature_range[-1]])
    return low, mid, high

feature_ranges = [np.linspace(min(data[:, i]), max(data[:, i]), 100) for i in range(data.shape[1])]
fuzzy_sets = [define_fuzzy_set(r) for r in feature_ranges]

# 3. 使用Wang-Mendel算法从数据中提取规则

def wang_mendel_algorithm(data, target):
    rules = {}
    for idx, sample in enumerate(data):
        rule_key = ()
        max_md = 0
        for i, feature_value in enumerate(sample):
            memberships = [
                (fuzz.interp_membership(feature_ranges[i], fuzzy_sets[i][0], feature_value), "Low"),
                (fuzz.interp_membership(feature_ranges[i], fuzzy_sets[i][1], feature_value), "Mid"),
                (fuzz.interp_membership(feature_ranges[i], fuzzy_sets[i][2], feature_value), "High")
            ]

            antecedent = max(memberships, key=lambda item: item[0])[1]
            rule_key += (antecedent, )

        if rule_key not in rules or max_md < max([md[0] for md in memberships]):
            rules[rule_key] = target[idx]

    return rules

rules = wang_mendel_algorithm(data, target)
print(rules)

def classify_sample(sample, rules):
    max_match_degree = -np.inf
    predicted_class = None

    for rule_key, rule_class in rules.items():
        match_degree = 1
        for i, feature_value in enumerate(sample):
            if rule_key[i] == "Low":
                membership = fuzz.interp_membership(feature_ranges[i], fuzzy_sets[i][0], feature_value)
            elif rule_key[i] == "Mid":
                membership = fuzz.interp_membership(feature_ranges[i], fuzzy_sets[i][1], feature_value)
            elif rule_key[i] == "High":
                membership = fuzz.interp_membership(feature_ranges[i], fuzzy_sets[i][2], feature_value)

            match_degree *= membership

        if match_degree > max_match_degree:
            max_match_degree = match_degree
            predicted_class = rule_class

    return predicted_class

# 定义5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for train_index, test_index in kf.split(data):
    # 划分训练集和测试集
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]

    # 使用Wang-Mendel算法从训练数据中提取规则
    rules = wang_mendel_algorithm(X_train, y_train)

    # 使用规则对测试数据进行分类
    predictions = [classify_sample(sample, rules) for sample in X_test]

    # 计算并存储每次的准确率
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)
    print("Fold accuracy:", accuracy)

# 计算平均准确率
average_accuracy = np.mean(accuracies)
print("Average accuracy:", average_accuracy)

