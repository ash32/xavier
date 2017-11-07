import matplotlib.pyplot as plt
from collections import Iterable


def plot_portfolio_values(portf_vals):
    if not isinstance(portf_vals, Iterable):
        portf_vals = [portf_vals]

    for s in portf_vals:
        plt.plot(s)
    plt.show()
