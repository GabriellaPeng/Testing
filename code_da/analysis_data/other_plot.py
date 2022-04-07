from matplotlib.pyplot import plot as plt

def fill_between(x_data, y_data_lower, y_data_upper, color=(0.73,0.21,0.41), alpha=0.6, step='mid'):
    plt.fill_between(x_data, y_data_lower, y_data_upper,
                    color=color, alpha=alpha, step=step)

def subplots(fig, nrow, ncol, index, **kwargs):
    plt.subplot(nrow, ncol, index, **kwargs)

