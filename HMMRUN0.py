# %% [markdown]
# # HMM Analysis for Andreev State Spectroscopy
# 
# This notebook provides functionality for analyzing Andreev state spectroscopy data
# using Hidden Markov Models (HMM).

# %%
%load_ext autoreload
%autoreload 2


# %%
import quasiparticleFunctions as qp
from hmm_utils import (create_physics_based_transition_matrix,
                          get_means_covars)
from hmm_analysis import HMMAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import seaborn as sns
from hmm_workflows import *

# %% [markdown]
# ## Single Power at Fixed Flux Workflow

# %%
# Initialize analyzer
data_dir = "/Users/shanto/LFL/HMM_Benchmarking/data/"
analyzer = HMMAnalyzer(data_dir, num_modes=2)

# %%
# Load data files
analyzer.load_data_files("phi_0p450")

# %%
# Process data for specific attenuation
atten = 26
analyzer.load_and_process_data(atten=atten)

# %%
%matplotlib qt

# %%
# Get initial parameters
result = get_means_covars(analyzer.data, analyzer.num_modes)
means_guess = result['means']
covars_guess = result['covariances']

# %%
means_guess = analyzer.model.means_
covars_guess = analyzer.model.covars_

# %%
# Initialize and fit model
analyzer.initialize_model(means_guess, covars_guess)
analyzer.fit_model()

# %%
# Decode states and calculate probabilities
logprob, states = analyzer.decode_states()
mean_occ, probs = analyzer.calculate_occupation_probabilities(states)
print(f"Mean occupation: {mean_occ}")
print(f"Probabilities: {probs}")

# Save all results and plots
analyzer.save_analysis_results(states, atten, means_guess, covars_guess)

# Calculate SNRs
snrs = analyzer.calculate_snrs()
print("SNRs:", snrs)

# %% [markdown]
# ## Variable Power at Fixed Flux (Bootstrapping) Workflow

# %%
data_dir = "/Users/shanto/LFL/HMM_Benchmarking/data/"
analyzer = HMMAnalyzer(data_dir, num_modes=2)
bootstrapping_analysis(analyzer, "phi_0p450", non_linear_atten=26, snr_threshold=1.5)

# %% [markdown]
# ## Variable Power and Flux (Bootstrapping) Workflow: *Human-in-the-loop*

# %%
data_dir = "/Users/shanto/LFL/HMM_Benchmarking/data/"
analyze_flux_sweep(data_dir, num_modes=2, non_linear_atten=26, snr_threshold=1.5)

# %% [markdown]
# ## Variable Power and Flux (Bootstrapping) Workflow: *Human-at-Instantiation*
# 

# %%
data_dir = "/Users/shanto/LFL/HMM_Benchmarking/data/"
init_param_dir = "/Users/shanto/LFL/HMM_Benchmarking/initial_params/"
non_linear_attens = [22]

# %%
phi_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith("phi_") and os.path.isdir(os.path.join(data_dir, d))])

# %%
create_and_save_initial_params(data_dir, phi_dirs, non_linear_attens, num_modes=3, save_dir=init_param_dir)

# %%
analyze_flux_sweep_auto(data_dir, num_modes=3, phi_dirs=phi_dirs, non_linear_attens=non_linear_attens, snr_threshold=1.5, init_param_dir=init_param_dir)


