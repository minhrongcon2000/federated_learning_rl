import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(reward_means, reward_stdds, n_clients, figure_dir):
    reward_means = np.array(reward_means)
    reward_stdds = np.array(reward_stdds)
    plt.plot(np.arange(len(reward_means)), reward_means)
    plt.fill_between(np.arange(len(reward_means)), 
                    reward_means + 1.96 * reward_stdds / np.sqrt(n_clients),
                    reward_means - 1.96 * reward_stdds / np.sqrt(n_clients),
                    alpha=0.2)
    plt.savefig(figure_dir, dpi=300)
    plt.show()