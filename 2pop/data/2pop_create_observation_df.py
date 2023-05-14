from helpers import helper_funcs
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import h5py

from elephant.statistics import fanofactor, isi, cv, mean_firing_rate
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from neo import SpikeTrain


def calculate_summary_statistics(sample, time_start=500):

    # Lists to store sumstats for each spiketrain
    fire_rate_inhibitory = []
    fire_rate_excitatory = []
    fano_inhibitory = []
    fano_excitatory = []
    interspike_inhibitory = []
    interspike_excitatory = []
    cv_inhibitory = []
    cv_excitatory = []

    # Get spiketrains from 100 excitatory and 100 inhibitory neurons
    in_spiketrains = sample["spiketrains_L4I"]
    ex_spiketrains = sample["spiketrains_L4E"]

    # Add individual spiketrains to lists
    for ex_st, in_st in zip(ex_spiketrains.values(), in_spiketrains.values()):
        # Slice
        sliced_in_st = np.array([inh for inh in in_st[()] if inh > time_start]) - time_start
        sliced_ex_st = np.array([ex for ex in ex_st[()] if ex > time_start]) - time_start

        # Convert numpy spiketrains to neo.SpikeTrain
        neo_spiketrain_inhibitory = SpikeTrain(sliced_in_st, t_stop=2000, units="ms")
        neo_spiketrain_excitatory = SpikeTrain(sliced_ex_st, t_stop=2000, units="ms")

        # Firing rate
        fire_rate_inhibitory.append(mean_firing_rate(neo_spiketrain_inhibitory)) \
            if len(neo_spiketrain_inhibitory) > 0 else fire_rate_inhibitory.append(0)
        fire_rate_excitatory.append(mean_firing_rate(neo_spiketrain_excitatory)) \
            if len(neo_spiketrain_excitatory) > 0 else fire_rate_excitatory.append(0)

        # All spiketrains, used in fanofactor
        fano_inhibitory.append(neo_spiketrain_inhibitory)
        fano_excitatory.append(neo_spiketrain_excitatory)

        # interspike interal
        interspike_inhibitory.append(np.mean(isi(neo_spiketrain_inhibitory))) if \
            len(neo_spiketrain_inhibitory) > 1 else interspike_inhibitory.append(np.nan)
        interspike_excitatory.append(np.mean(isi(neo_spiketrain_excitatory))) if \
            len(neo_spiketrain_excitatory) > 1 else interspike_excitatory.append(np.nan)

        # CV
        cv_inhibitory.append(cv(isi(in_st))) if len(neo_spiketrain_inhibitory) > 1 \
            else cv_inhibitory.append(np.nan)
        cv_excitatory.append(cv(isi(ex_st))) if len(neo_spiketrain_excitatory) > 1 \
            else cv_excitatory.append(np.nan)

    return {"mean_firing_rate_inhibitory": np.mean(fire_rate_inhibitory),
            "mean_firing_rate_excitatory": np.mean(fire_rate_excitatory),
            "fanofactor_inhibitory": fanofactor(fano_inhibitory),
            "fanofactor_excitatory": fanofactor(fano_excitatory),
            "mean_interspike_interval_inhibitory": np.nanmean(interspike_inhibitory),
            "mean_interspike_interval_excitatory": np.nanmean(interspike_excitatory),
            "mean_cv_inhibitory": np.nanmean(cv_inhibitory),
            "mean_cv_excitatory": np.nanmean(cv_excitatory)}


def generate_dicts(two_pop_path):
    """Store each file (sample) in a dict overview."""
    samples = {}

    sample_num = 0
    # Loop through files in folder
    for filename in two_pop_path.glob("*.h5"):
        file = h5py.File(filename, "r")

        # Loop through values (samples) inside each file
        for sample in file.values():
            samples[f"sample_{sample_num}"] = dict(sample)

            sample_num += 1

    return samples


def reduce_bin_size(histogram, bin_size, time_start=500):
    """Reduces bin size of histograms"""

    # Define new histogram with number of channels and new length
    temp_histogram = np.zeros((histogram.shape[0], histogram.shape[1] // bin_size))
    new_histogram = np.zeros((histogram.shape[0], (2500-time_start)))  # total time - time start

    # compute new histogram by summing values in each bin
    for i in range(temp_histogram.shape[0]):
        for j in range(temp_histogram.shape[1]):
            # Create temporarily new histogram with reduced bin size
            temp_histogram[i, j] = np.sum(histogram[i, j * bin_size:(j + 1) * bin_size])
            # Slice temp histogram as we want to start at 500 ms
            new_histogram[i] = temp_histogram[i][time_start:]

    return new_histogram


def generate_histogram_sumstat(hist_samples, time_start=500):
    # Shape sizes of tensor
    bin_size = 10  # reduce length of histogram to 1/10 (keeping all info)
    sample_size = len(hist_samples)  # len([h for h in hist_samples.values() if "histograms" in h]) # todo:
    channels = hist_samples["sample_0"]["histograms"].shape[0]
    channel_size = hist_samples["sample_0"]["histograms"].shape[1] // bin_size

    # Create placeholder
    numpy_placeholder = np.zeros((sample_size, channels, channel_size-time_start))

    i = 0
    for sample in hist_samples.values():
        # if "histograms" in list(sample.keys()):  # todo: delete when all samples
        compressed_histogram = reduce_bin_size(sample["histograms"][()], bin_size=bin_size)
        numpy_placeholder[i] = np.stack(compressed_histogram, axis=0)
        i += 1

    # Convert to torch tensor (to be able to use with sbi)
    tensor = torch.tensor(numpy_placeholder, dtype=torch.float32)

    return tensor


def generate_df(samples):
    """Generates dataframe of parameters and summary statistics"""
    df_dict = {"g_ee": [],
               "g_ei": [],
               "g_ie": [],
               "g_ii": [],
               "mean_firing_rate_inhibitory": [],
               "mean_firing_rate_excitatory": [],
               "fanofactor_inhibitory": [],
               "fanofactor_excitatory": [],
               "mean_interspike_interval_inhibitory": [],
               "mean_interspike_interval_excitatory": [],
               "mean_cv_inhibitory": [],
               "mean_cv_excitatory": []}

    # Add values from samples and store in dict
    for sample_dict in samples.values():
        # if "summary_statistics" in list(sample_dict.keys()):  # todo: delete this when full samples
        # Get varying parameters and sumstats
        df_dict["g_ee"].append(dict(sample_dict["network_params"].attrs.items())["g_YX"][0][0])
        df_dict["g_ei"].append(dict(sample_dict["network_params"].attrs.items())["g_YX"][0][1])
        df_dict["g_ie"].append(dict(sample_dict["network_params"].attrs.items())["g_YX"][1][0])
        df_dict["g_ii"].append(dict(sample_dict["network_params"].attrs.items())["g_YX"][1][1])

        sumstat_dict = calculate_summary_statistics(sample_dict)

        df_dict["mean_firing_rate_inhibitory"].append(sumstat_dict["mean_firing_rate_inhibitory"])
        df_dict["mean_firing_rate_excitatory"].append(sumstat_dict["mean_firing_rate_excitatory"])
        df_dict["fanofactor_inhibitory"].append(sumstat_dict["fanofactor_inhibitory"])
        df_dict["fanofactor_excitatory"].append(sumstat_dict["fanofactor_excitatory"])
        df_dict["mean_interspike_interval_inhibitory"].append(sumstat_dict["mean_interspike_interval_inhibitory"])
        df_dict["mean_interspike_interval_excitatory"].append(sumstat_dict["mean_interspike_interval_excitatory"])
        df_dict["mean_cv_inhibitory"].append(sumstat_dict["mean_cv_inhibitory"])
        df_dict["mean_cv_excitatory"].append(sumstat_dict["mean_cv_excitatory"])

    return pd.DataFrame(data=df_dict)


def create_datasets(load_pop, save_folder):
    # Get data from files
    samples = generate_dicts(load_pop)

    # Store in dataframe
    df_full = generate_df(samples)

    # Create observation
    df_obs = pd.DataFrame(df_full.to_dict())

    # Generate tensor of histogram sumstat
    obs_sumstat_hist = generate_histogram_sumstat(samples)

    # Save dfs with generated sumstats and corresponding parameters
    helper_funcs.save_file(
        df_obs, save_folder.joinpath("2pop_observation_samples"))

    # Save datasets with histogram as sumstat
    helper_funcs.save_file(
        obs_sumstat_hist, save_folder.joinpath("2pop_observation_histogram_samples"))


def df_final_obs(df):

    final_obs_dict = {"g_ee": df.iloc[0]["g_ee"],
                      "g_ei": df.iloc[0]["g_ei"],
                      "g_ie": df.iloc[0]["g_ie"],
                      "g_ii": df.iloc[0]["g_ii"]}

    final_obs_dict.update({"mean_firing_rate_inhibitory": df["mean_firing_rate_inhibitory"].mean()})
    final_obs_dict.update({"mean_firing_rate_excitatory": df["mean_firing_rate_excitatory"].mean()})
    final_obs_dict.update({"fanofactor_inhibitory": df["fanofactor_inhibitory"].mean()})
    final_obs_dict.update({"fanofactor_excitatory": df["fanofactor_excitatory"].mean()})
    final_obs_dict.update({"mean_interspike_interval_inhibitory": df["mean_interspike_interval_inhibitory"].mean()})
    final_obs_dict.update({"mean_interspike_interval_excitatory": df["mean_interspike_interval_excitatory"].mean()})
    final_obs_dict.update({"mean_cv_inhibitory": df["mean_cv_inhibitory"].mean()})
    final_obs_dict.update({"mean_cv_excitatory": df["mean_cv_excitatory"].mean()})

    return pd.DataFrame(final_obs_dict, index=[0])


if __name__ == "__main__":
    # Save and load paths
    save_pop_path = Path(f"./obs")
    load_pop_path = Path(f"../../../simulation_data/2pop/observation_data/")

    # Create observation df
    create_datasets(load_pop=load_pop_path, save_folder=save_pop_path)

    # Load save file (since I don't want to fix setup)
    df_obs_samples = helper_funcs.load_file(Path("obs/2pop_observation_samples"))
    obs_samples_hist = helper_funcs.load_file(Path("obs/2pop_observation_histogram_samples"))

    ## Creating final observation
    # df_obs: mean of each summary statistics
    df_obs = df_final_obs(df_obs_samples)

    # histogram_obs: Take the sample closest to the observed dataframe
    scaler = MinMaxScaler()
    df_obs_samples_scaled = scaler.fit_transform(df_obs_samples.to_numpy())
    df_obs_scaled = scaler.transform(df_obs.to_numpy())

    # distances = cdist(df_obs_samples.to_numpy(), df_obs.to_numpy())
    # closest_distance_idx = np.argmin(distances)
    distances = cdist(df_obs_samples_scaled, df_obs_scaled)
    closest_distance_idx = np.argmin(distances)
    print(closest_distance_idx)

    histogram_obs = obs_samples_hist[closest_distance_idx]


    # Save files
    helper_funcs.save_file(df_obs, "./2pop_observation")
    helper_funcs.save_file(histogram_obs, "./2pop_observation_histogram")
    helper_funcs.save_file(closest_distance_idx, "./2pop_closest_distance_idx")
