import numpy as np
import ipdb
from matplotlib import colors
import matplotlib.pyplot as plt

def plot_dist1D(data_array, label, step=10, **kwargs):
    """

    :param data_array:
    :param label:
    :param step:
    :param kwargs:
    :return:
    """
    lower = data_array.min()
    upper = data_array.max()

    x = np.linspace(lower, upper, step)
    interval = (upper-lower)/(step-1)

    y = np.zeros(x.size)

    for data in data_array:
        i = int((data-lower)/(interval+0.00000001))
        y[i]+=1

    x = x+interval/2

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()

    plt.plot(x, y, color='blue', alpha=0.3)
    plt.xlabel(label)

    return fig



