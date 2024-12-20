import pandas as pd


if __name__ == '__main__':
    data_set_path = 'Datasets/sachs.csv'
    variable_to_be_changed = 'pkc'
    output_file_path = 'Datasets/sachs_' + variable_to_be_changed + '.csv'
    # 读取CSV文件
    df = pd.read_csv(data_set_path)

    # 计算'pip3'列的中位数
    median_of_changed_variable = df[variable_to_be_changed].median()

    # 根据中位数进行二分类
    df['Class'] = df[variable_to_be_changed].apply(lambda x: 1 if x > median_of_changed_variable else 0)

    # 重新排列列的顺序，将'Class'列移动到最后一列
    columns = list(df.columns)
    columns.remove('Class')
    columns.append('Class')
    df = df[columns]

    # 保存结果到CSV文件
    df.to_csv(output_file_path, index=False)
