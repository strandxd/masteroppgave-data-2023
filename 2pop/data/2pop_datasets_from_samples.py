from helpers import helper_funcs
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import h5py

from elephant.statistics import fanofactor, isi, cv, mean_firing_rate
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


def generate_histogram_sumstat(hist_samples, dropped_indexes, time_start=500,
                               obs_index=None, create_obs=False):
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

    if create_obs:
        # Slice simulation up to and past observation, then concatenate
        sim_sumstat = torch.cat((tensor[:obs_index], tensor[obs_index+1:]), dim=0)
        obs_sumstat = tensor[obs_index]

        # Drop samples that were dropped from df
        sim_sumstat = torch.index_select(sim_sumstat, dim=0,
                                         index=torch.tensor([i for i in range(sim_sumstat.size(0))
                                                             if i not in dropped_indexes]))

        return sim_sumstat, obs_sumstat
    else:
        # Drop samples that were dropped from df
        tensor = torch.index_select(tensor, dim=0,
                                    index=torch.tensor([i for i in range(tensor.size(0))
                                                        if i not in dropped_indexes]))
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


def create_datasets(sim_num, load_pop, save_folder, method=None, seed=5555):
    print("Loading samples")
    # Get data from files
    samples = generate_dicts(load_pop)
    print("Finished loading samples\n")

    print("Generating df")
    # Store in dataframe
    df_full = generate_df(samples)
    dropped_indexes = df_full.index.difference(df_full.dropna().index)
    # df_full = df_full.dropna().reset_index(drop=True)
    print(dropped_indexes)
    print("Finished generating df\n")

    print("Creating histograms")
    # Create observation data from initial simulation
    if sim_num == 0:
        np.random.seed(seed)
        rand_int = np.random.randint(0, len(df_full) + 1)

        assert rand_int not in list(dropped_indexes), "Observed sample index is dropped"

        df_obs = pd.DataFrame(df_full.iloc[rand_int].to_dict(), index=[0])
        df_sim = df_full.drop(index=rand_int, axis=0).reset_index(drop=True)  # drop obs
        df_sim = df_sim.dropna().reset_index(drop=True)  # drop nan

        # Generate tensor of histogram sumstat (both obs and simulations)
        sim_sumstat_hist, obs_sumstat_hist = generate_histogram_sumstat(samples,
                                                                        dropped_indexes,
                                                                        obs_index=rand_int,
                                                                        create_obs=True)

        # Save dfs with generated sumstats and corresponding parameters
        helper_funcs.save_file(
            df_obs, save_folder.joinpath("obs/2pop_observation_base"))
        helper_funcs.save_file(
            df_sim,
            save_folder.joinpath(f"sim_{sim_num}/2pop_simulations_{sim_num}"))

        # Save datasets with histogram as sumstat
        helper_funcs.save_file(
            obs_sumstat_hist, save_folder.joinpath("obs/2pop_observation_histogram_base"))
        helper_funcs.save_file(
            sim_sumstat_hist,
            save_folder.joinpath(f"sim_{sim_num}/2pop_simulations_histogram_{sim_num}"))

    else:
        # Generate tensor of histogram sumstat
        sim_sumstat_hist = generate_histogram_sumstat(samples, dropped_indexes, create_obs=False)

        # Store to folder
        helper_funcs.save_file(
            df_full, save_folder.joinpath(f"sim_{sim_num}/{method}/2pop_simulations_{sim_num}"))

        helper_funcs.save_file(
            sim_sumstat_hist,
            save_folder.joinpath(f"sim_{sim_num}/{method}/2pop_simulations_histogram_{sim_num}"))


if __name__ == "__main__":
    # Change these
    simulation_number = 2
    method = "sbi_embedding"  # sbi_simple, sbi_embedding or abc_regression

    # Save and load paths
    save_pop_path = Path(f"./")
    if simulation_number == 0:
        load_pop_path = Path(f"../../../simulation_data/2pop/simulation_{simulation_number}/")
        #load_pop_path = Path(f"../../../simulation_data/2pop/simulation_test/")

        # Create datasets
        create_datasets(sim_num=simulation_number, load_pop=load_pop_path,
                        save_folder=save_pop_path, method=None)

    else:
        load_pop_path = Path(f"../../../simulation_data/2pop/simulation_{simulation_number}/{method}/")

        # Create datasets
        create_datasets(sim_num=simulation_number, load_pop=load_pop_path,
                        save_folder=save_pop_path, method=method)
