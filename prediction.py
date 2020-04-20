import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl


def read_sunspots(file_name):
    with open(file_name, 'r') as f:
        all_sunspots = [row.replace('\n', '').split('\t') for row in f.readlines()]
        return [[int(sunspot[0]), int(sunspot[1])] for sunspot in all_sunspots]


def prepare_data(_data, _number_of_columns):
    _x = []
    _y = []

    for i in range(_number_of_columns, len(sunspots)):
        temp_array = []
        for j in range(_number_of_columns, 0, -1):
            temp_array.append(sunspots[i - j][1])

        _x.append(temp_array)
        _y.append(sunspots[i][1])

    return np.array(_x), np.array(_y).reshape(-1, 1)


def normalize_data(_data):
    min_number = min(_data)
    max_number = max(_data)

    low = -1
    high = 1

    return [(number - min_number) / np.floor(max_number - min_number) * (high - low) + low for number in _data]

sunspots = read_sunspots('sunspot.txt')

iterations = 10_000
columns = 4
P, T = prepare_data(sunspots, columns)
PU = P[:200]
TU = T[:200]
b = 1

np.random.seed(1)
w1 = (2 * np.random.random(columns) - 1).reshape(-1, 1)
# print(f"Neurono svorio koeficientai:\n{w1}")

input = P
target = T

net = nl.net.newp([[0, 1], [0, 1], [0, 1], [0, 1]], 1)
error = net.train(input, target, epochs=100, show=10, lr=0.000005)


plt.plot(error)
plt.xlabel('Epoch number')
plt.ylabel('Train error')
plt.grid()
plt.show()