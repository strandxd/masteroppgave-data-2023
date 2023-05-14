import numpy as np
import h5py, os, time

num_runners = 1
total_num_sims = 1
sim_dir = os.path.join(".", "simulation_test")
n_threads = 6

if not os.path.isdir(sim_dir):
    os.mkdir(sim_dir)

kernel_params = {
    'total_num_virtual_procs': n_threads,   # total number of virtual processes
    'resolution': 0.1,                      # time resolution [ms]
    'rng_type': 'mt19937',
    'print_time': False,
}

simulation_params = {"T_sim": 500.,
                     "n_threads": n_threads,
                     "n_record": 100,
                     }

# parameters which are fixed
network_params = {
        "population_names": np.array(["L4E".encode("ascii"), "L4I".encode("ascii")]),# Unicode not supported by h5py
        "N_X": np.array([21915, 5479]),
        "k_ext_X": np.array([2100, 1900]),
        "C_YX":  np.array([[0.050, 0.135],   # Connection probabilities. Element C_ij gives probabiity
                           [0.079, 0.160]]), # of connection FROM population j ONTO population j. E.g.
                                             # index [0,0] is from E onto E, element[1,0] is from E onto I.

        "d_E": 1.5,              # synaptic delay, excitatory [ms]
        "d_I": 0.75,             # synaptic delay, inhibitory [ms]
        "J": 87.81,              # reference synapyic strength [pA]
}

neuron_params = {             # Set parameters for iaf_psc_exp
        "tau_m": 10.0,        # membrance time constant [ms]
        "t_ref": 2.0,         # refractory period [ms]
        "C_m": 250.0,         # membrane capacitance [pF]
        "E_L": -65.0,         # resting membrane potential [mV]
        "V_th": -50.0,        # threshold potential [mV]
        "V_reset": -65.0,     # reset potential [mV]
        "tau_syn_ex": 0.5,    # excitatory synaptic time constant [ms]
        "tau_syn_in": 0.5     # inhibitory synaptic time constant [ms]
        }


def draw_parameters():
    ex_weight_min = 0.5
    ex_weight_max = 2.0
    in_weight_rel_min = 4.5
    in_weight_rel_max = 8.0
    rng = np.random.default_rng()
    g_ee = rng.uniform(low=ex_weight_min, high=ex_weight_max)
    g_ie = rng.uniform(low=ex_weight_min, high=ex_weight_max)
    g_ei = -rng.uniform(low=in_weight_rel_min, high=in_weight_rel_max) * g_ee
    g_ii = -rng.uniform(low=in_weight_rel_min, high=in_weight_rel_max) * g_ie
    g_YX = np.array([[g_ee, g_ei],
                     [g_ie, g_ii]])
    return {"g_YX": g_YX}

# divvy up simulations
sims_per_rank = total_num_sims // num_runners
remainder = total_num_sims % num_runners

for i in range(1, num_runners+1):
    num_sims = sims_per_rank if i >= remainder else sims_per_rank + 1
    print(f"Num sims: {num_sims}")
    rng = np.random.default_rng()
    seeds = rng.integers(low=1, high=2 ** (31), size=num_sims)

    with h5py.File(os.path.join(sim_dir, f"runner_{i:01d}.h5"), "w") as f:
        for j in range(num_sims):
            seed = seeds[j]
            varying_parameters = draw_parameters()
            simulation_params.update({"seed": seed})
            network_params.update(varying_parameters)
            grp = f.create_group(f"{j:05d}")
            subgrp1 = grp.create_group("kernel_params")
            for key, val in kernel_params.items():
                subgrp1.attrs.create(key, val)
            subgrp2 = grp.create_group("simulation_params")
            for key, val in simulation_params.items():
                subgrp2.attrs.create(key, val)
            subgrp3 = grp.create_group("neuron_params")
            for key, val in neuron_params.items():
                subgrp3.attrs.create(key, val)
            subgrp4 = grp.create_group("network_params")
            for key, val in network_params.items():
                subgrp4.attrs.create(key, val)

