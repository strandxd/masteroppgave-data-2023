from helpers import helper_funcs
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import copy
import os


def simple_abc_df(df_sim, df_obs, param_names, sumstat_names, q=0.1):
    # Summary statistics
    sumstat_sim = df_sim[sumstat_names].to_numpy()
    sumstat_obs = df_obs[sumstat_names].to_numpy()

    # Scale params
    sc = StandardScaler()
    std_sumstat_sim = sc.fit_transform(sumstat_sim)
    std_sumstat_obs = sc.transform(sumstat_obs)

    # Theta
    params_sim = df_sim[param_names].to_numpy()

    # Make sure this axis part is functional. axis=sumstat_sim.ndim-1 should work
    distances = np.linalg.norm((std_sumstat_sim - std_sumstat_obs), ord=2, axis=sumstat_sim.ndim-1)

    # Quantile (basically how restrictive ABC should be. I.e. 1.0 = include all, 0.0 = include none)
    epsilon = np.quantile(distances, q=q)
    is_accepted = distances <= epsilon

    # Filter out simulations that are not accepted
    params_sim = params_sim[is_accepted]
    sumstat_sim = sumstat_sim[is_accepted]

    # Create new dataframes based on accepted samples
    abc_params_df = pd.DataFrame(dict(zip(param_names, np.stack(params_sim, axis=-1))))
    abc_sumstat_df = pd.DataFrame(dict(zip(sumstat_names, np.stack(sumstat_sim, axis=-1))))
    abc_full_df = pd.concat([abc_params_df, abc_sumstat_df], axis=1)

    return abc_params_df, abc_full_df


def linreg_adj(df_abc, df_obs, param_names, sumstat_names):
    sumstat_obs = df_obs[sumstat_names].to_numpy()
    sumstat_sim = df_abc[sumstat_names].to_numpy()
    abc_params = df_abc[param_names].to_numpy()

    # Copy to correct variables
    X = copy.deepcopy(sumstat_sim)
    X_obs = copy.deepcopy(sumstat_obs).reshape(1, -1)
    y = copy.deepcopy(abc_params)

    # If I only use 1 sumstat. This wont ever happen (I hope).
    if X.ndim == 1:
        print("This shouldn't happen")
        X = X.reshape(-1, 1)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_obs = scaler.transform(X_obs)

    # Train model
    reg_model = LinearRegression(fit_intercept=True)
    reg_model.fit(X, y)

    # Can't log values as some are negative (Y matrix)
    params_adjusted = reg_model.predict(X_obs) + y - reg_model.predict(X)
    df_adjusted = pd.DataFrame(dict(zip(param_names, np.stack(params_adjusted, axis=-1))))
    df_adjusted_full = pd.concat([df_adjusted, df_abc_full[sumstat_names]], axis=1)

    return df_adjusted, df_adjusted_full


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
    simulation_number = 0

    # Load obs and sims
    obs_pop_path = Path("../data/2pop_observation")
    if simulation_number == 0:
        sim_pop_path = Path(f"../data/sim_{simulation_number}/2pop_simulations_{simulation_number}")
    else:
        sim_pop_path = Path(f"../data/sim_{simulation_number}/abc_regression/2pop_simulations_{simulation_number}")
    df_observation = helper_funcs.load_file(obs_pop_path)
    df_simulations = helper_funcs.load_file(sim_pop_path)

    # Save paths
    save_abc_simple_path = Path(f"./abc_data/sim_{simulation_number}/simple/")
    save_abc_regression_path = Path(f"./abc_data/sim_{simulation_number}/regression/")
    # Create paths
    if not save_abc_simple_path.exists():
        save_abc_simple_path.mkdir(parents=True)
    if not os.path.exists(save_abc_regression_path):
        save_abc_regression_path.mkdir(parents=True)

    # Create simple abc dataframes
    df_abc_params, df_abc_full = simple_abc_df(df_simulations,
                                               df_observation,
                                               parameter_names,
                                               summary_stat_names,
                                               q=0.1)

    # Create adjusted abc dataframes
    df_abc_adjusted_params, df_abc_adjusted_full = linreg_adj(df_abc_full,
                                                              df_observation,
                                                              parameter_names,
                                                              summary_stat_names)

    # save simple abc to file (full contains summary statistics)
    helper_funcs.save_file(
        df_abc_params,
        save_abc_simple_path.joinpath(f"2pop_df_abc_simple_params_{simulation_number}")
    )
    helper_funcs.save_file(
        df_abc_full,
        save_abc_simple_path.joinpath(f"2pop_df_abc_simple_full_{simulation_number}")
    )

    # save adjusted abc to file (adjusted with linear regression)
    helper_funcs.save_file(
        df_abc_adjusted_params,
        save_abc_regression_path.joinpath(f"2pop_df_abc_regression_params_{simulation_number}")
    )
    helper_funcs.save_file(
        df_abc_adjusted_full,
        save_abc_regression_path.joinpath(f"2pop_df_abc_regression_full_{simulation_number}")
    )
