from fuzzy_system import *


def encode_labels(df):
    '''
    Covert string labels to discrete labels.
    :param df:
    :return:
    '''
    le = LabelEncoder()
    ledf = copy.deepcopy(df)
    ledf.iloc[:, -1] = le.fit_transform(ledf.iloc[:, -1])
    return ledf, le.classes_


def load_data(data_set_path):
    original_data = pd.read_csv(data_set_path)
    labels = original_data.iloc[:,-1]
    original_y = labels.to_numpy()
    le_data, le_original_classes = encode_labels(original_data)
    scalar = MinMaxScaler()
    le_x = le_data.iloc[:, :-1]
    le_y = le_data.iloc[:, -1]
    y = le_y.values
    normalised_x = scalar.fit_transform(le_x)

    return normalised_x, y, original_y

def load_CD_data(data_set_path):
    original_data = pd.read_csv(data_set_path)
    labels = original_data.iloc[:,-1]
    original_y = labels.to_numpy()
    le_data, le_original_classes = encode_labels(original_data)
    scalar = MinMaxScaler()
    le_x = le_data.iloc[:, :-1]
    le_y = le_data.iloc[:, -1]
    y = le_y.values
    normalised_x = scalar.fit_transform(le_x)

    return le_data, normalised_x, y, original_y

def load_sachs_data(data_set_path):
    original_sachs = pd.read_csv(data_set_path)
    scalar = MinMaxScaler()
    normalised_sachs = scalar.fit_transform(original_sachs)
    return normalised_sachs