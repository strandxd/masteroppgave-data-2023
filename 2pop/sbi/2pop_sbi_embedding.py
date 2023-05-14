from pathlib import Path
from sbi.inference import SNPE
from scipy import stats
from sbi import utils
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler

import numpy as np
import pandas as pd
import torch.nn as nn

import torch.nn.functional as F

import torch
import optuna
import sys

# to enable running in cmd
sys.path.append("../../")
from helpers import helper_funcs


class CNNEmbedding(nn.Module):
    """
    CNN used as an embedding net in sbi.
    Note: Takes for granted a batch size of 256 in the comments below.
    Shape: <batch_size>, <channels>, <channel_length>
    """

    def __init__(self, num_sumstats):
        super(CNNEmbedding, self).__init__()

        # convolution:
        # from shape (256x2x2000) -- (256x4x2000)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1,
                               padding="same")

        # max pooling layers:
        # from shape (256x4x2000) -- (256x4x500)
        # from shape (256x4x2000) -- (256x4x1000)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # convolution:
        # from shape (256x4x1000) -- (256x8x1000)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1,
                               padding="same")

        # max pooling layer:
        # from shape (256x8x1000) -- (256x8x125)
        # from shape (256x8x1000) -- (256x8x500)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # fully connected
        self.fc1 = nn.Linear(in_features=8 * 500, out_features=350)
        self.fc2 = nn.Linear(in_features=350, out_features=num_sumstats)

    def forward(self, x):
        # convolution
        x = self.conv1(x)
        x = F.relu(x)

        # max pooling
        x = self.pool1(x)

        # convolution
        x = self.conv2(x)
        x = F.relu(x)

        # max pooling
        x = self.pool2(x)

        # Fully connected
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


def sbi_embedding_optimization(trial,
                               prior_assumption,
                               df_sim,
                               sim_sumstat_hist,
                               param_names,
                               max_epochs):
    """


    Parameters
    ----------
    trial : optuna.Trial
        optuna object
    prior_assumption : np.array, optional
        distribution used for simulations (priors for sim_0)
    df_sim : pd.DataFrame
        param values for synapse strength. Use df_sim[param_names] to access parameters
    sim_sumstat_hist : torch.tensor
        sumstats (histogram)
    param_names : list
        list of parameter names
    max_epochs : int
        maximum number of epochs

    Returns
    -------
    float
        average rmspe (as metric for optimization)

    """
    # Parameters to optimize
    hidden_features = trial.suggest_int("hidden_features", 32, 256)
    num_transforms = trial.suggest_int("num_transforms", 2, 14)
    num_sumstats = trial.suggest_int("num_sumstats", 5, 70)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.01, log=True)

    # Define embedding net and nn to create posteriors
    embedding_net = CNNEmbedding(num_sumstats=num_sumstats)
    neural_posterior = utils.posterior_nn(model='maf',
                                          embedding_net=embedding_net,
                                          hidden_features=hidden_features,
                                          num_transforms=num_transforms,
                                          learning_rate=learning_rate,
                                          z_score_theta="independent",
                                          z_score_x="structured")

    inference = SNPE(prior=prior_assumption, density_estimator=neural_posterior)

    # Append simulations (histograms)
    sim_params = torch.from_numpy(df_sim[param_names].to_numpy(np.float32))
    inference.append_simulations(sim_params, sim_sumstat_hist)

    # # Train density estimator
    inference.train(training_batch_size=512,
                    stop_after_epochs=30,
                    max_num_epochs=max_epochs)

    return inference.summary['best_validation_log_prob'][0]


def run_best_sbi_embedding(best_params,
                           prior_assumption,
                           df_sim,
                           sim_sumstat_hist,
                           obs_sumstat_hist,
                           param_names,
                           max_epochs,
                           num_samples=2000):
    # Define embedding net and nn to create posteriors
    embedding_net = CNNEmbedding(num_sumstats=best_params["num_sumstats"])
    neural_posterior = utils.posterior_nn(model='maf',
                                          embedding_net=embedding_net,
                                          hidden_features=best_params["hidden_features"],
                                          num_transforms=best_params["num_transforms"],
                                          learning_rate=best_params["learning_rate"],
                                          z_score_theta="independent",
                                          z_score_x="structured")

    inference = SNPE(prior=prior_assumption, density_estimator=neural_posterior)

    # Append simulations (histograms)
    sim_params = torch.from_numpy(df_sim[param_names].to_numpy(np.float32))
    # scaled_sim_sumstat_hist = minmax_scale(sim_sumstat_hist, axis=1)  # scaled to 0-1 range
    inference.append_simulations(sim_params, sim_sumstat_hist)

    # Train estimator
    density_estimator = inference.train(training_batch_size=512,
                                        stop_after_epochs=50,
                                        max_num_epochs=max_epochs)

    # Create dfs
    posterior_generator = inference.build_posterior(density_estimator=density_estimator)
    posterior = posterior_generator.set_default_x(obs_sumstat_hist).sample((num_samples,))
    sbi_embedding_params_df = pd.DataFrame(posterior, columns=param_names)

    return sbi_embedding_params_df, posterior_generator


if __name__ == "__main__":
    # Names of parameters and summary statistics
    parameter_names = ["g_ee", "g_ie", "g_ei", "g_ii"]
    summary_stat_names = ["mean_firing_rate_inhibitory",
                          "mean_firing_rate_excitatory",
                          "fanofactor_inhibitory",
                          "fanofactor_excitatory",
                          "mean_interspike_interval_inhibitory",
                          "mean_interspike_interval_excitatory",
                          "mean_cv_inhibitory",
                          "mean_cv_excitatory"]
    simulation = 1

    ## Load datasets
    # Observations (both with generated sumstats and histogram as sumstat)
    df_observation = helper_funcs.load_file(Path("../data/2pop_observation"))
    sumstat_observation_hist = helper_funcs.load_file(Path("../data/2pop_observation_histogram"))

    # Simulations (both with generated sumstats and histogram as sumstat)
    if simulation == 0:
        df_simulations = helper_funcs.load_file(
            Path(f"../data/sim_{simulation}/2pop_simulations_{simulation}"))
        sumstat_simulation_hist = helper_funcs.load_file(
            Path(f"../data/sim_{simulation}/2pop_simulations_histogram_{simulation}"))
    else:
        df_simulations = helper_funcs.load_file(
            Path(f"../data/sim_{simulation}/sbi_embedding/2pop_simulations_{simulation}"))
        sumstat_simulation_hist = helper_funcs.load_file(
            Path(f"../data/sim_{simulation}/sbi_embedding/2pop_simulations_histogram_{simulation}"))

    # todo: this scales both channels equally
    ## Min-max scale histograms
    # Scale simulations
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    # for i in range(sumstat_simulation_hist.shape[0]):
    #     # Make one long sequence to scale
    #     temp = sumstat_simulation_hist[i].reshape((-1, 1))
    #
    #     # Scale and reshape back
    #     sumstat_simulation_hist[i] = torch.tensor(scaler.fit_transform(temp).reshape((2, 2000)))

    # Scale observation
    temp_obs = sumstat_observation_hist.reshape((-1, 1))
    sumstat_observation_hist = torch.tensor(scaler.fit_transform(temp_obs).reshape((2, 2000)))

    # # todo: this scales both channels individually
    # # Min-max scale histograms
    # # Scale simulations
    # for i in range(sumstat_simulation_hist.shape[0]):
    #     sumstat_simulation_hist[i] = torch.tensor(minmax_scale(sumstat_simulation_hist[i], axis=1))
    #
    # # Scale observation
    # sumstat_observation_hist = torch.tensor(minmax_scale(sumstat_observation_hist, axis=1))

    # Priors from previous simulation (uniform prior for simulation 0)
    if simulation == 0:
        # Order: g_ee, g_ie, g_ei, g_ii
        prior_min = [0.5, 0.5, -16.0, -16.0]
        prior_max = [2.0, 2.0, -2.25, -2.25]

        prior_simulated_distribution = \
            utils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))
    else:
        generator_model = helper_funcs.load_file(
            Path(f"./sbi_data/sim_{simulation-1}/embedding/2pop_embedding_generator_model_{simulation-1}"))
        df_sbi_previous = helper_funcs.load_file(
            Path(f"./sbi_data/sim_{simulation-1}/embedding/2pop_df_sbi_embedding_params_{simulation-1}"))

        # Generate model attributes
        generator_model.mean = torch.mean(generator_model.sample((2000,)), dim=0)
        generator_model.variance = torch.var(generator_model.sample((2000, )), dim=0)
        lower_bound = torch.tensor((df_sbi_previous["g_ee"].min(),
                                    df_sbi_previous["g_ie"].min(),
                                    df_sbi_previous["g_ei"].min(),
                                    df_sbi_previous["g_ii"].min()))

        upper_bound = torch.tensor((df_sbi_previous["g_ee"].max(),
                                    df_sbi_previous["g_ie"].max(),
                                    df_sbi_previous["g_ei"].max(),
                                    df_sbi_previous["g_ii"].max()))

        prior_simulated_distribution, _, _ = \
            utils.user_input_checks.process_prior(generator_model,
                                                  {"lower_bound": lower_bound,
                                                   "upper_bound": upper_bound})

    # Do the optimization
    print("starting optimization")
    n_trials = 50
    max_epochs = 800
    num_samples = 2000
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial:
                   sbi_embedding_optimization(trial,
                                              prior_assumption=prior_simulated_distribution,
                                              df_sim=df_simulations,
                                              sim_sumstat_hist=sumstat_simulation_hist,
                                              param_names=parameter_names,
                                              max_epochs=max_epochs),
                   n_trials=n_trials)
    print("finished optimization")

    # Store best params and metric (logprob)
    best_parameters = study.best_params
    best_metric = study.best_value

    print(f"Best params: {best_parameters}")
    print(f"Best metric: {best_metric}")

    # Train based on best params
    df_sbi_embedding_params, sbi_posterior_generator_model = run_best_sbi_embedding(
        best_params=best_parameters,
        prior_assumption=prior_simulated_distribution,
        df_sim=df_simulations,
        sim_sumstat_hist=sumstat_simulation_hist,
        obs_sumstat_hist=sumstat_observation_hist,
        param_names=parameter_names,
        max_epochs=1000,
        num_samples=num_samples
    )

    # Save samples from posterior distribution
    save_posterior_sbi = Path(f"./sbi_data/sim_{simulation}/embedding/")
    if not save_posterior_sbi.exists():
        save_posterior_sbi.mkdir(parents=True)

    print(f"Saving files to path: {save_posterior_sbi}")
    # save df
    helper_funcs.save_file(df_sbi_embedding_params,
                           save_posterior_sbi.joinpath(
                               f"2pop_df_sbi_embedding_params_{simulation}"))

    # save model (used as prior in next iteration)
    helper_funcs.save_file(sbi_posterior_generator_model,
                           save_posterior_sbi.joinpath(
                               f"2pop_embedding_generator_model_{simulation}"))

    # Save best params and metric
    helper_funcs.save_file(best_parameters,
                           save_posterior_sbi.joinpath(
                               f"2pop_best_params_embedding_sim_{simulation}"))

    helper_funcs.save_file(best_metric,
                           save_posterior_sbi.joinpath(
                               f"2pop_best_metric_embedding_sim_{simulation}"))
