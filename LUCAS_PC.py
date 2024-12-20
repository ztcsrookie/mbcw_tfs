

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import numpy as np

import csv

if __name__ == '__main__':
    file_path = 'Datasets/sachs.csv'
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=float)
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        labels = next(reader)

    print(data.shape)
    print(labels)

    graphs = {}
    graphs_nx = {}
    labels = [f'{col}' for i, col in enumerate(labels)]
    cg = pc(data)

    pyd = GraphUtils.to_pydot(cg.G, labels=labels)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    pyd = GraphUtils.to_pydot(cg.G, labels=labels)
    CG_save_path = 'Results/CGs/sachs_PC.png'
    pyd.write_png(CG_save_path)