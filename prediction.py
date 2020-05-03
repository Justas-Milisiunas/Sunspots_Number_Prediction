import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from Graphs import *
# import neurolab as nl

np.random.seed(1)

sunspots = read_sunspots('sunspot.txt')
# show_sunspots_activity_plot(sunspots)
# show_input_output_plot(sunspots)

data_min = min(np.array(sunspots)[:, 1])
data_max = max(np.array(sunspots)[:, 1])

dataset = prepare_data(sunspots, 10)

training_set = dataset[:200]
validation_set = dataset[:200]

n_inputs = len(training_set[0]) - 1
n_outputs = 1

network = NeuralNetwork(n_inputs, 1, n_outputs, data_min, data_max)
progress = network.train(training_set, 0.1, 1000)

# plt.plot(np.arange(len(progress)), progress)
# plt.title('Epochų klaidų grafikas')
# plt.xlabel('Epocha')
# plt.ylabel('Klaida')
# plt.show()

MSE, MAD, errors, predictions = network.validate(dataset)
#
# show_errors_histogram(errors)
# # print(errors)
# show_errors_plot(errors)
show_model_verification_plot(predictions, title='Modelio verifikacija 1700-2014 metais')
# print(f"MSE: {MSE} MAD: {MAD}")