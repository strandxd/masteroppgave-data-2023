import pickle
import ast
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


def save_file(file, filename):
    """Save a file using pickle"""
    with open(filename, mode="wb") as write_file:
        pickle.dump(file, write_file)


def load_file(filename):
    """Load file using pickle"""
    with open(filename, mode="rb") as read_file:
        file = pickle.load(read_file)

    return file


def calculate_rmspe(true_param, pred_param):
    """Calculates root mean squared percentage error"""
    rmspe = np.sqrt(np.mean(np.square((true_param - pred_param) / true_param)))
    return rmspe * 100


# Not really using it, got it in notebook
def plot_prior(min_val, max_val, x_label, ax, N=10000, save_plot=False, filepath=None):
    """Plots prior distribution of the first simulation (uniform)"""
    # Min and max value of synapse weight (from github simulation)

    scale = np.abs((min_val + max_val)) * 0.2

    # Distribution calculator
    # todo remove: Distributions (ee = exhitatory to exhitatory, ie = exhitatory to inhibitory etc.)
    distribution = stats.uniform(loc=min_val, scale=(max_val - min_val))

    # weight_values to evaluate distribution
    xs = np.linspace(min_val - scale, max_val + scale, num=N)

    # Save plot
    if save_plot:
        # todo: remove or make clean
        if not filepath:
            print(f"Need to specify path. Current selected path: {filepath}")
            return

        plt.figure()
        plt.plot(xs, distribution.pdf(xs))
        plt.fill_between(xs, distribution.pdf(xs), alpha=.5)
        plt.xlabel(x_label)
        plt.savefig(f"{filepath}.pdf", format="pdf")

    else:
        # Make different colors
        if x_label == "g_ee" or x_label == "g_ie":
            ax.plot(xs, distribution.pdf(xs))
            ax.fill_between(xs, distribution.pdf(xs), alpha=.5)
            ax.set_xlabel(x_label)
        else:
            ax.plot(xs, distribution.pdf(xs), color="C1")
            ax.fill_between(xs, distribution.pdf(xs), color="C1", alpha=.5)
            ax.set_xlabel(x_label)


# Not really using it, got it in notebook
def plot_posterior(df, param_name, df_obs, ax, bbox=(1.25, 1.0), borderpad=1, N=10000,
                   show_ylabel=False, show_legend=True, save_plot=False, filepath=None):
    """Plot posterior distribution"""
    posterior_distribution = stats.gaussian_kde(df[param_name])
    # todo: could do posterior_distribution.resample(#number) - would this be correct?

    # True theta value
    theta_true = float(df_obs[param_name])

    # kdeplot
    kde_xaxis = sns.kdeplot(data=df,
                            x=param_name,
                            fill=True,
                            color="C1",
                            label="Posterior",
                            ax=ax).get_xaxis()

    # Get data to for vertical lines to illustrate (for now) mode (MAP - maximum a posteriori)
    # and true theta
    x_min, x_max = kde_xaxis.get_data_interval()[0], kde_xaxis.get_data_interval()[1]
    xs = np.linspace(x_min, x_max, num=N, endpoint=True)

    # Mark most dense estimated point and true theta
    point_estimate_y = np.max(posterior_distribution.pdf(xs))
    point_estimate_x = xs[np.argmax(posterior_distribution.pdf(xs))]
    y_max_true = posterior_distribution.pdf(theta_true)

    if save_plot:
        if not filepath:
            print(f"Need to specify path. Current selected path: {filepath}")
            return

        # Create new figure
        plt.figure()
        # Plot again
        sns.kdeplot(data=df,x=param_name,fill=True,color="C1",label="Posterior", ax=ax)
        # Plot vertical lines
        plt.vlines(x=theta_true, ymin=0, ymax=y_max_true * 0.99, color="black",
                   label=rf"$\theta_{{true}}$: {theta_true:.2}")
        plt.vlines(x=point_estimate_x, ymin=0, ymax=point_estimate_y, ls=":", color="purple",
                   label=rf"$\hat{{\theta}}_{{est}}$: {point_estimate_x:.3}")

        plt.legend(loc="upper right", bbox_to_anchor=bbox)

        if not show_ylabel:
            plt.ylabel(None)

        if not show_legend:
            plt.legend().remove()

        # Saving figure
        print(f"Saving to file: {filepath}.pdf")
        plt.savefig(f"{filepath}.pdf", format="pdf")
        print("File saved.")

    else:
        # Plot vertical lines
        ax.vlines(x=theta_true, ymin=0, ymax=y_max_true * 0.99, color="black",
                  label=rf"$\theta_{{true}}$: {theta_true:.2}")
        ax.vlines(x=point_estimate_x, ymin=0, ymax=point_estimate_y, ls=":", color="purple",
                  label=rf"$\hat{{\theta}}_{{est}}$: {point_estimate_x:.3}")

        ax.legend(loc="upper right", bbox_to_anchor=bbox)

        if not show_ylabel:
            ax.set_ylabel(None)

        if not show_legend:
            ax.get_legend().remove()

