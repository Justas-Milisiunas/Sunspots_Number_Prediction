import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from Graphs import *
# import neurolab as nl

sunspots = read_sunspots('sunspot.txt')
# show_input_output_plot(sunspots)

data_min = min(np.array(sunspots)[:, 1])
data_max = max(np.array(sunspots)[:, 1])

dataset = prepare_data(sunspots, 2)

training_set = dataset[:200]
validation_set = dataset[200:]

n_inputs = len(training_set[0]) - 1
n_outputs = 1

network = NeuralNetwork(n_inputs, 1, n_outputs, data_min, data_max)
network.train(training_set, 0.01, 10000)

MSE, MAD, errors, predictions = network.validate(validation_set)

# show_errors_histogram(errors)
# show_errors_plot(errors)
show_model_verification_plot(predictions, title='Modelio verifikacija 1900-2014 metais')
print(f"MSE: {MSE} MAD: {MAD}")

# def read_sunspots(file_name):
#     with open(file_name, 'r') as f:
#         all_sunspots = [row.replace('\n', '').split('\t') for row in f.readlines()]
#         return [[int(sunspot[0]), int(sunspot[1])] for sunspot in all_sunspots]
#
#
# def prepare_data(_data, _number_of_columns):
#     _x = []
#     _y = []
#
#     for i in range(_number_of_columns, len(sunspots)):
#         temp_array = []
#         for j in range(_number_of_columns, 0, -1):
#             temp_array.append(sunspots[i - j][1])
#
#         _x.append(temp_array)
#         _y.append(sunspots[i][1])
#
#     return np.array(_x), np.array(_y).reshape(-1, 1)
#
#
# def normalize_data(_data):
#     min_number = min(_data)
#     max_number = max(_data)
#
#     low = -1
#     high = 1
#
#     return [(number - min_number) / np.floor(max_number - min_number) * (high - low) + low for number in _data]
#
#
# sunspots = read_sunspots('sunspot.txt')
#
# iterations = 10_000
# columns = 4
# P, T = prepare_data(sunspots, columns)
# PU = P[:200]
# TU = T[:200]
# b = 1
#
# np.random.seed(1)
# w1 = (2 * np.random.random(columns) - 1).reshape(-1, 1)
# # print(f"Neurono svorio koeficientai:\n{w1}")
#
# input = PU
# target = TU
#
#
# net = nl.net.newp([[0, 1], [0, 1], [0, 1], [0, 1]], 1)
# net.layers[0].transf = nl.trans.PureLin()
# net.errorf = nl.error.MSE()
# error = net.trainf(input=input, target=target, epochs=1000, show=100, lr=0.00001, net=net)
# tt = net.sim(input=P[200:])
# correct_results = T[200:]
# sum = 0
# max_error = 0
# for i in range(len(tt)):
#     error = np.abs(correct_results[i] - tt[i])
#     if error >= max_error:
#         max_error = error
#     sum += np.abs(correct_results[i] - tt[i])
#
# print(sum)
# print(sum/len(correct_results))
# print(max_error)
#
# plt.plot(error)
# plt.xlabel('Epoch number')
# plt.ylabel('Train error')
# plt.grid()
# plt.show()
