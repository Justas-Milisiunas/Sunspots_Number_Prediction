import numpy as np
# from temp import *
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def show_model_verification_plot(_data, title):
    _data = np.array(_data)
    plt.plot(np.arange(1700, 1700 + len(_data)), _data[:, 0], color='r', label='Spėjimas')
    plt.plot(np.arange(1700, 1700 + len(_data)), _data[:, 1], color='b', label='Tikėtasis')

    plt.title(title)
    plt.xlabel('Metai')
    plt.ylabel('Saulės dėmių skaičiųs')
    plt.legend()
    plt.show()


def show_sunspots_activity_plot(_sunspots):
    """
    Shows sunspots activity diagram
    :param _sunspots: sunspots data
    """
    plt.plot([sunspot[0] for sunspot in _sunspots], [sunspot[1] for sunspot in _sunspots])
    plt.title("Saulės dėmių aktyvumo 1700-2014 m. grafikas")
    plt.xlabel('Metai')
    plt.ylabel('Saulės dėmių sk.')
    plt.show()


def show_input_output_plot(_sunspots):
    """
    Shows 3d scatter plot with input, output data (Assuming using two previous years to predict)
    :param _sunspots: sunspots data
    """
    data = np.array(prepare_data(_sunspots, 2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])

    ax.set_title('Įvesčių ir išvesčių trimatė diagrama')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()


def show_errors_histogram(_errors):
    plt.hist(_errors, weights=np.ones(len(_errors)), bins=10, ec='black')

    plt.title('Prognozės klaidų histograma')
    plt.xlabel('Paklaida')
    plt.ylabel('Dažnis')
    plt.show()


def show_errors_plot(_errors):
    plt.plot(np.arange(1700, 1700 + len(_errors)), _errors)

    plt.title('Prognozės klaidų diagrama')
    plt.xlabel('Metai')
    plt.ylabel('Paklaida')
    plt.show()


# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
def prepare_data(_data, _number_of_columns):
    _x = []
    _y = []

    for i in range(_number_of_columns, len(_data)):
        temp_array = []
        for j in range(_number_of_columns, 0, -1):
            temp_array.append(_data[i - j][1])

        _x.append(temp_array)
        _y.append(_data[i][1])

    x = list(_x)
    y = list(_y)

    for i, number in enumerate(y):
        x[i].append(number)

    return x


def read_sunspots(file_name):
    with open(file_name, 'r') as f:
        all_sunspots = [row.replace('\n', '').split('\t') for row in f.readlines()]
        return [[int(sunspot[0]), int(sunspot[1])] for sunspot in all_sunspots]