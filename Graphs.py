import numpy as np
# from temp import *
from NeuralNetwork import NeuralNetwork


# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
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

    low = 0
    high = 1

    return [(number - min_number) / (max_number - min_number) * (high - low) + low for number in
            _data], min_number, max_number


def unnormalize_data(_data, _min, _max):
    return [number * (_max - _min) + _min for number in _data]


def unnormalize_number(_x, _min, _max):
    return _x * (_max - _min) + _min


def read_sunspots(file_name):
    with open(file_name, 'r') as f:
        all_sunspots = [row.replace('\n', '').split('\t') for row in f.readlines()]
        return [[int(sunspot[0]), int(sunspot[1])] for sunspot in all_sunspots]


sunspots = read_sunspots('sunspot.txt')
data_min = min(np.array(sunspots)[:, 1])
data_max = max(np.array(sunspots)[:, 1])

# numbers = [number[1] for number in sunspots]
# normalized_numbers, min_number, max_number = normalize_data(numbers)
# for i, number in enumerate(normalized_numbers):
#     sunspots[i][1] = number
x, y = prepare_data(sunspots, 2)
x = list(x)
y = list(y)
for i in range(len(x)):
    for j in range(len(x[i])):
        x[i] = list(x[i])
for i, number in enumerate(y):
    x[i].append(number[0])

dataset = x[:200]
n_inputs = len(dataset[0]) - 1
n_outputs = 1

network = NeuralNetwork(n_inputs, 1, n_outputs, data_min, data_max)
network.train(dataset, 0.001, 50000)

errors = 0
max_error = 0
for row in dataset:
    prediction = network.predict(row)
    expected_output = row[-1]
    # print('Expected=%d, Got=%d' % (row[-1], prediction))
    # denormalized_expected = unnormalize_number(expected_output, data_min, data_max)
    denormalized_expected = expected_output
    # denormalized_prediction = unnormalize_number(prediction, data_min, data_max)
    denormalized_prediction = prediction

    error = np.abs(denormalized_expected - denormalized_prediction)
    if error >= max_error:
        max_error = error
    errors += (denormalized_expected - denormalized_prediction) ** 2
    print(f"Expected={denormalized_expected} Expected normal={expected_output} Got={denormalized_prediction} Got normal={prediction}")

print(f"MSE: {errors / len(dataset)} Errors average: {errors / len(x[200:])} Max error: {max_error}")

# np.random.seed(1)
# network = initialize_network(n_inputs, 1, n_outputs)
# print(network)
# print("\n\n\n")
# train_network(network, dataset, 0.01, 10000, n_outputs)
# print(network)
# errors = 0
# max_error = 0
# for row in x[200:]:
#     prediction = predict(network, row)
#     prediction = unnormalize_number(prediction, min_number, max_number)
#     expected_output = unnormalize_number(row[-1], min_number, max_number)
#     # print('Expected=%d, Got=%d' % (row[-1], prediction))
#     error = np.abs(expected_output - prediction)
#     if error >= max_error:
#         max_error = error
#     errors += (expected_output - prediction)**2
#     print(f"Expected={expected_output} Got={prediction}")
#
# print(f"Errors sum: {errors/len(x[200:])} Errors average: {errors / len(x[200:])} Max error: {max_error}")

# plt.plot([sunspot[0] for sunspot in sunspots], [sunspot[1] for sunspot in sunspots])
# plt.title("Saulės dėmių aktyvumo 1700-2014 m. grafikas")
# plt.xlabel('Metai')
# plt.ylabel('Saulės dėmių sk.')
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # for i in range(columns):
# #     ax.scatter(x[:, i], y)
#
# # ax.scatter(x[:, 0], x[:, 1], x[:, 2], x[:, 3], y)
# ax.scatter(P[:, 0], P[:, 1], T)
# plt.show()
#
# import neurolab as nl
#
# # Logical &
# input = [[0, 0], [0, 1], [1, 0], [1, 1]]
# target = [[0], [0], [0], [1]]
#
# # Create net with 2 inputs and 1 neuron
# net = nl.net.newp([[0, 1], [0, 1]], 1)
#
# # train with delta rule
# # see net.trainf
# error = net.trainf(input, target, epochs=100, show=10, lr=0.1)
#
# # Plot results
# import pylab as pl
#
# pl.plot(error)
# pl.xlabel('Epoch number')
# pl.ylabel('Train error')
# pl.grid()
# pl.show()
