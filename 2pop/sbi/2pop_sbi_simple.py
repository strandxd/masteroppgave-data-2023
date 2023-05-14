from sbi import utils
from sbi.inference import SNPE
from pathlib import Path
from scipy import stats

import numpy as np
import pandas as pd

import torch
import optuna
import sys

# to enable running in cmd
sys.path.append("../../")
from helpers import helper_funcs


def sbi_simple_optimization(trial,
                            prior_assumption,
                            df_sim,
                            param_names,
                            sumstat_names,
                            max_epochs):
    # Parameters to optimize
    hidden_features = trial.suggest_int("hidden_features", 32, 256)
    num_transforms = trial.suggest_int("num_transforms", 2, 14)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.01, log=True)

    # Define nn to create posteriors
    neural_posterior = utils.posterior_nn(model='maf',
                                          hidden_features=hidden_features,
                                          num_transforms=num_transforms,
                                          learning_rate=learning_rate,
                                          z_score_theta="independent",
                                          z_score_x="independent")

    inference = SNPE(prior=prior_assumption, density_estimator=neural_posterior)

    # Append simulations (normal sumstats)
    sim_params = torch.from_numpy(df_sim[param_names].to_numpy(np.float32))
    sim_sumstat = torch.from_numpy(df_sim[sumstat_names].to_numpy(np.float32))
    inference.append_simulations(sim_params, sim_sumstat)

    inference.train(training_batch_size=512,
                    stop_after_epochs=30,
                    max_num_epochs=max_epochs)

    return inference.summary['best_validation_log_prob'][0]
    #return np.mean(total_rmspe)


def run_best_sbi_simple(best_params,
                        prior_assumption,
                        df_sim,
                        df_obs,
                        param_names,
                        sumstat_names,
                        max_epochs,
                        num_samples=5000):
    # Define neural posterior
    neural_posterior = utils.posterior_nn(model="maf",
                                          hidden_features=best_params["hidden_features"],
                                          num_transforms=best_params["num_transforms"],
                                          learning_rate=best_params["learning_rate"],
                                          z_score_theta="independent",
                                          z_score_x="independent")

    inference = SNPE(prior=prior_assumption, density_estimator=neural_posterior)

    # Append simulations (normal sumstats)
    sim_params = torch.from_numpy(df_sim[param_names].to_numpy(np.float32))
    sim_sumstats = torch.from_numpy(df_sim[sumstat_names].to_numpy(np.float32))
    inference.append_simulations(sim_params, sim_sumstats)

    # Observed summary statistics
    x_obs = torch.from_numpy(df_obs[sumstat_names].to_numpy(np.float32))

    # Train estimator
    density_estimator = inference.train(training_batch_size=512,
                                        stop_after_epochs=50,
                                        max_num_epochs=max_epochs)

    # Create dfs
    posterior_generator = inference.build_posterior(density_estimator=density_estimator)
    posterior = posterior_generator.set_default_x(x_obs).sample((num_samples,))
    sbi_simple_params_df = pd.DataFrame(posterior, columns=param_names)

    return sbi_simple_params_df, posterior_generator


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

    # Load datasets
    df_observation = helper_funcs.load_file(Path(f"../data/2pop_observation"))
    if simulation == 0:
        df_simulations = helper_funcs.load_file(
            Path(f"../data/sim_{simulation}/2pop_simulations_{simulation}"))
    else:
        df_simulations = helper_funcs.load_file(
            Path(f"../data/sim_{simulation}/sbi_simple/2pop_simulations_{simulation}"))

    # Priors from previous simulation (uniform prior for simulation 0)
    # todo: discuss if I should always pass uniform priors to sbi or update for further sims
    if simulation == 0:
        # Order: g_ee, g_ie, g_ei, g_ii
        prior_min = [0.5, 0.5, -16.0, -16.0]
        prior_max = [2.0, 2.0, -2.25, -2.25]

        prior_simulated_distribution = \
            utils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))
    else:
        # Posteriors from past simulation becomes prior
        generator_model = helper_funcs.load_file(
            Path(f"./sbi_data/sim_{simulation-1}/simple/2pop_simple_generator_model_{simulation-1}"))
        prior_simulated_distribution, _, _ = utils.user_input_checks.process_prior(generator_model)

    print("Starting optimization")
    n_trials = 50
    max_epochs = 800
    num_samples = 2000
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial:
                   sbi_simple_optimization(trial,
                                           prior_assumption=prior_simulated_distribution,
                                           df_sim=df_simulations,
                                           param_names=parameter_names,
                                           sumstat_names=summary_stat_names,
                                           max_epochs=max_epochs),
                   n_trials=n_trials)
    print("finished optimization")

    # Store best params and rmspe
    best_parameters = study.best_params
    best_metric = study.best_value

    print(f"Best params: {best_parameters}")
    print(f"Best metric: {best_metric}")

    # Train based on best params
    df_sbi_simple_params, sbi_posterior_generator_model = run_best_sbi_simple(
        best_params=best_parameters,
        prior_assumption=prior_simulated_distribution,
        df_sim=df_simulations,
        df_obs=df_observation,
        param_names=parameter_names,
        sumstat_names=summary_stat_names,
        max_epochs=1000,
        num_samples=num_samples
    )

    # Save samples from posterior distribution
    save_posterior_sbi = Path(f"./sbi_data/sim_{simulation}/simple/")
    if not save_posterior_sbi.exists():
        save_posterior_sbi.mkdir(parents=True)

    print(f"Saving files to path: {save_posterior_sbi}")
    # save df
    helper_funcs.save_file(df_sbi_simple_params,
                           save_posterior_sbi.joinpath(f"2pop_df_sbi_simple_params_{simulation}"))
    # save model (used as prior in next iteration)
    helper_funcs.save_file(sbi_posterior_generator_model,
                           save_posterior_sbi.joinpath(f"2pop_simple_generator_model_{simulation}"))

    # Save best params and metric
    helper_funcs.save_file(best_parameters,
                           save_posterior_sbi.joinpath(f"2pop_best_params_simple_sim_{simulation}"))

    helper_funcs.save_file(best_metric,
                           save_posterior_sbi.joinpath(f"2pop_best_metric_simple_sim_{simulation}"))
