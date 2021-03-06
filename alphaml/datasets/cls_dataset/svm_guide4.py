import numpy as np


def load_svmguide4(data_folder):
    L = []
    file_path = data_folder + 'svmguide4'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\n')[0].split(' ')
            d = [0] * 11
            label = int(items[0]) - 1
            d[0] = label if label >= 0 else label + 7
            for item in items[1:]:
                key, value = item.split(':')
                d[int(key)] = float(value)
            L.append(d)
        data = np.array(L)
        return data[:, 1:], data[:, 0]
