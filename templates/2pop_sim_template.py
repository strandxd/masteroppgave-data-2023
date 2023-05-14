import copy
import fcntl
import hashlib
import os
import sys
import time

import h5py
import nest
import numpy as np

# Summary statistic imports
from neo.core import SpikeTrain
from elephant.statistics import mean_firing_rate, fanofactor, isi, cv

n_record = 100     # number of neurons to record spikes from
path = "./simulation_test/"


def run_2pop_network(savefile, sim):
    with h5py.File(savefile, "r") as f:
        kernel_params = dict(f[sim]["kernel_params"].attrs)
        simulation_params = dict(f[sim]["simulation_params"].attrs)
        neuron_params = dict(f[sim]["neuron_params"].attrs)
        network_params = dict(f[sim]["network_params"].attrs)

    # for some reason, nest.set() does not accept np.bool_
    kernel_params["print_time"] = bool(kernel_params["print_time"])

    nest.ResetKernel()
    nest.set(rng_seed=simulation_params["seed"], **kernel_params)
    nest.print_time = True
    nest.SetDefaults("iaf_psc_exp", neuron_params)

    pops = []
    spike_recorders = []
    for i, N in enumerate(network_params["N_X"]):
        pop = nest.Create('iaf_psc_exp', N)
        pop.V_m = nest.random.uniform(min=neuron_params["V_reset"],
                                      max=neuron_params["V_th"])
        # 0.008 is a fixed constant (see Table 5 in [1], 8 per s -> 0.008 per ms)
        pop.I_e = network_params["J"] * neuron_params["tau_syn_ex"] * \
            0.008 * network_params["k_ext_X"][i]
        pops.append(pop)
        spike_recorder = nest.Create("spike_recorder")
        nest.Connect(pop, spike_recorder)
        spike_recorders.append(spike_recorder)

    # i postsynaptic
    for i in range(2):
        # j presynaptic
        for j in range(2):
            # Connect local populations
            presyn_type = "E" if "E" in str(
                network_params["population_names"][j]) else "I"  # Avoid re-encoding?
            syn_spec = {"synapse_model": "static_synapse",
                        "weight": network_params["g_YX"][i][j] * network_params["J"],
                        "delay": nest.math.redraw(
                            nest.random.normal(
                                mean= network_params["d_" + presyn_type],
                                std=(network_params["d_" + presyn_type] * 0.5)
                                ),
                            min=nest.resolution - 0.5 * nest.resolution,
                            max=np.Inf)
                            }


            conn_spec = {"rule": "fixed_indegree",
                         "indegree": int(network_params["C_YX"][i][j] * network_params["N_X"][j])}

            nest.Connect(pops[j], pops[i], syn_spec=syn_spec,
                         conn_spec=conn_spec)

    print(nest.GetDefaults())
    print("Starting simulation", flush=True)
    nest.Simulate(simulation_params["T_sim"])
    print("Fininshed simulation", flush=True)

    print("Collecting spikes", flush=True)
    histograms = []
    spiketrains = []
    for j, sr in enumerate(spike_recorders):
        events = nest.GetStatus(sr, "events")[0]
        histogram, _ = np.histogram(events["times"], bins=np.arange(
            0, simulation_params["T_sim"] + 0.2, 0.1) - 0.05)
        histograms.append(histogram)
        spiketrains.append([])
        neuron_indices = pops[j].tolist()
        print(f"length of neuron indices: {len(neuron_indices)}")
        # todo: need to change to this
        for i in range(len(pops[j])):
        #for i in range(len(neuron_indices)):
        # Now creates all spiketrains
            spikes = events["times"][events["senders"] == neuron_indices[i]]
            spiketrains[-1].append(spikes)

    histograms = np.array(histograms)
    print("Finished collecting spikes", flush=True)

    print("Saving to file")

    with h5py.File(savefile, "r+") as f:
        f[sim].create_dataset(name="histograms", data=histograms)
        f[sim].create_dataset(name="summary_statistics",
                              data=str(create_sum_stats(spiketrains, simulation_params["T_sim"], simulation_params)))
        spiketrains_L4E = f[sim].create_group(name="spiketrains_L4E")
        spiketrains_L4I = f[sim].create_group(name="spiketrains_L4I")

        for i in range(n_record):
            spiketrains_L4E.create_dataset(name=str(i), data=spiketrains[0][i])
            spiketrains_L4I.create_dataset(name=str(i), data=spiketrains[1][i])


def create_sum_stats(spiketrain_list, sim_time_end, sim_params):
    """Create summary statistics from neo.spiketrains"""
    decimals = 7
    print(f"len of spiketrain_list: {len(spiketrain_list)}")

    fire_rate = []
    spiketrains = []
    interspike_interval = []
    cv_list = []

    for pop_spiketrains in spiketrain_list:
        print(f"len of neuron spiketrains: {len(pop_spiketrains)}")
        for neuron_spiketrain in pop_spiketrains:
            spiketrain = SpikeTrain(neuron_spiketrain, t_stop=sim_time_end, units="ms")
            spiketrains.append(spiketrain)

            # Adding neuron statistics to list
            fire_rate.append(mean_firing_rate(spiketrain)) if len(neuron_spiketrain) > 0 \
                else fire_rate.append(0)
            interspike_interval.append(float(np.mean(isi(spiketrain)))) if len(neuron_spiketrain) > 1 \
                else interspike_interval.append(np.nan)
            cv_list.append(cv(isi(spiketrain))) if len(neuron_spiketrain) > 1 \
                else cv_list.append(np.nan)

    # Calculate some summary statistics
    return {"mean_fire_rate": np.round(np.mean(fire_rate), decimals=decimals),
            "fanofactor": np.round(fanofactor(spiketrains), decimals=decimals),
            "mean_interspike_interval": np.round(np.nanmean(interspike_interval),
                                                 decimals=decimals),
            "mean_cv": np.round(np.nanmean(cv_list), decimals=decimals)}


if __name__ == "__main__":
    runner_id = 1#int(sys.argv[1])
    print(f"runner id: {runner_id}")
    savefile = os.path.join(path, f"runner_{runner_id:01d}.h5")
    with h5py.File(savefile, "r") as f:
        sims = list(f.keys())
    for i, sim in enumerate(sims):
        run_2pop_network(savefile, sim)
