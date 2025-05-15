import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import quasiparticleFunctions as qp
from hmm_analysis import HMMAnalyzer
from hmm_utils import create_physics_based_transition_matrix, get_means_covars


def analyze_flux_sweep(data_dir, num_modes, non_linear_atten, snr_threshold):
    # Find all phi_* directories
    phi_dirs = sorted(
        [d for d in os.listdir(data_dir) if d.startswith("phi_") and os.path.isdir(os.path.join(data_dir, d))]
    )
    print(f"Found phi directories: {phi_dirs}")

    for phi_string in phi_dirs:
        print(f"\n=== Running analysis for {phi_string} ===")
        analyzer = HMMAnalyzer(data_dir, num_modes=num_modes)
        try:
            bootstrapping_analysis(analyzer, phi_string, non_linear_atten, snr_threshold)
        except Exception as e:
            print(f"Error processing {phi_string}: {e}")

def bootstrapping_analysis(analyzer, phi_string, non_linear_atten, snr_threshold):
    analyzer.load_data_files(phi_string)
    attenuations = analyzer.attenuations
    idx = np.where(attenuations == non_linear_atten)[0][0]
    means_guess = None
    covars_guess = None

    results = []  # Collect results for summary plotting

    for i in range(idx, len(attenuations)):
        atten = attenuations[i]
        print(f"\n=== Processing attenuation: {atten} dB ===")
        analyzer.load_and_process_data(atten=atten)
        
        # Get initial parameters for first run, or use previous model's
        if means_guess is None or covars_guess is None:
            result = get_means_covars(analyzer.data, analyzer.num_modes)
            means_guess = result['means']
            covars_guess = result['covariances']
        
        analyzer.initialize_model(means_guess, covars_guess)
        analyzer.fit_model()
        logprob, states = analyzer.decode_states()
        mean_occ, probs = analyzer.calculate_occupation_probabilities(states)
        analyzer.save_analysis_results(states, atten, means_guess, covars_guess)
        snrs = analyzer.calculate_snrs()
        rates = analyzer.calculate_transition_rates()
        
        # Store results
        results.append({
            "atten": atten,
            "mean_occ": mean_occ,
            "probs": probs,
            "snrs": snrs,
            "rates": rates
        })
        
        print(f"Mean occupation: {mean_occ}")
        print(f"Probabilities: {probs}")
        print("SNRs:", snrs)
        
        # Stop if any SNR is below threshold
        if (snrs[0] < snr_threshold):
            print(f"Stopping: SNR below threshold ({snr_threshold})")
            break
        
        # Use current model's means and covariances for next run
        means_guess = analyzer.model.means_
        covars_guess = analyzer.model.covars_

    # Call the summary plotting function
    savepath = analyzer.data_dir
    plot_bootstrap_summary(results, analyzer.num_modes, save_path=savepath)

def plot_bootstrap_summary(results, num_modes, save=True, save_path=None):
    atts = [r["atten"] for r in results]
    mean_occs = [r["mean_occ"] for r in results]
    probs = np.array([r["probs"] for r in results])  # shape: (n, num_modes)
    snrs = np.array([r["snrs"] for r in results])    # shape: (n, num_pairs)
    rates = np.array([r["rates"] for r in results])  # shape: (n, num_modes, num_modes)

    # Plot mean occupation
    plt.figure()
    plt.plot(atts, mean_occs, marker='o')
    plt.xlabel("Attenuation (dB)")
    plt.ylabel("Mean Occupation")
    plt.title("Mean Occupation vs Attenuation")
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(save_path, "mean_occupation.png"))
    plt.show()


    # Plot probabilities
    plt.figure()
    for i in range(num_modes):
        plt.plot(atts, probs[:, i], marker='o', label=f"State {i}")
    plt.xlabel("Attenuation (dB)")
    plt.ylabel("Probability")
    plt.title("State Probabilities vs Attenuation")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(save_path, "state_probabilities.png"))
    plt.show()

    # Plot SNRs
    plt.figure()
    for i in range(snrs.shape[1]):
        plt.plot(atts, snrs[:, i], marker='o', label=f"SNR {i}")
    plt.xlabel("Attenuation (dB)")
    plt.ylabel("SNR")
    plt.title("SNRs vs Attenuation")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(save_path, "snrs.png"))
    plt.show()

    # Plot transition rates
    plt.figure()
    for i in range(num_modes):
        for j in range(num_modes):
            if i != j:
                plt.plot(atts, rates[:, i, j], marker='o', label=f"{i}→{j}")
    plt.xlabel("Attenuation (dB)")
    plt.ylabel("Transition Rate (MHz)")
    plt.title("Transition Rates vs Attenuation")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(save_path, "transition_rates.png"))
    plt.show()

    # Plot transition rates
    plt.figure()
    for i in range(num_modes):
        for j in range(num_modes):
            if i != j:
                plt.plot(atts, rates[:, i, j], marker='o', label=f"{i}→{j}")
    plt.xlabel("Attenuation (dB)")
    plt.ylabel("Transition Rate (MHz)")
    plt.title("Transition Rates vs Attenuation")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(save_path, "transition_rates_log.png"))
    plt.show()

def save_initial_params(phi_dir, non_linear_atten, means, covars, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fname = f"init_params_{phi_dir}_DA{non_linear_atten}.npz"
    np.savez(os.path.join(save_dir, fname), means=means, covars=covars)
    print(f"Saved initial params for {phi_dir}, DA={non_linear_atten} to {fname}")

def load_initial_params(phi_dir, non_linear_atten, save_dir):
    fname = f"init_params_{phi_dir}_DA{non_linear_atten}.npz"
    data = np.load(os.path.join(save_dir, fname))
    return data['means'], data['covars']

def create_and_save_initial_params(data_dir, phi_dirs, non_linear_attens, num_modes, save_dir):
    """
    For each (phi_dir, non_linear_atten) pair, load the data, let the user select means/covars,
    and save them to disk for later automated use.
    """
    assert len(phi_dirs) == len(non_linear_attens), "phi_dirs and non_linear_attens must have the same length!"
    os.makedirs(save_dir, exist_ok=True)
    
    for phi_dir, atten in zip(phi_dirs, non_linear_attens):
        print(f"\n=== Processing {phi_dir} at DA={atten} ===")
        analyzer = HMMAnalyzer(data_dir, num_modes=num_modes)
        analyzer.load_data_files(phi_dir)
        analyzer.load_and_process_data(atten=atten)
        
        # User selects means and covars (e.g., using a helper or notebook cell)
        # This is where you can plot or inspect the data if needed
        result = get_means_covars(analyzer.data, analyzer.num_modes)
        means_guess = result['means']
        covars_guess = result['covariances']
        
        # Save to disk
        fname = f"init_params_{phi_dir}_DA{atten}.npz"
        np.savez(os.path.join(save_dir, fname), means=means_guess, covars=covars_guess)
        print(f"Saved: {fname}")

def bootstrapping_analysis_auto(analyzer, phi_string, non_linear_atten, snr_threshold, init_param_dir):
    analyzer.load_data_files(phi_string)
    attenuations = analyzer.attenuations
    idx = np.where(attenuations == non_linear_atten)[0][0]
    means_guess, covars_guess = load_initial_params(phi_string, non_linear_atten, init_param_dir)

    results = []
    for i in range(idx, len(attenuations)):
        atten = attenuations[i]
        print(f"\n=== Processing attenuation: {atten} dB ===")
        analyzer.load_and_process_data(atten=atten)
        analyzer.initialize_model(means_guess, covars_guess)
        analyzer.fit_model()
        logprob, states = analyzer.decode_states()
        mean_occ, probs = analyzer.calculate_occupation_probabilities(states)
        analyzer.save_analysis_results(states, atten, means_guess, covars_guess)
        snrs = analyzer.calculate_snrs()
        rates = analyzer.calculate_transition_rates()
        results.append({
            "atten": atten,
            "mean_occ": mean_occ,
            "probs": probs,
            "snrs": snrs,
            "rates": rates
        })
        print(f"Mean occupation: {mean_occ}")
        print(f"Probabilities: {probs}")
        print("SNRs:", snrs)
        if snrs[0] < snr_threshold:
            print(f"Stopping: SNR below threshold ({snr_threshold})")
            break
        means_guess = analyzer.model.means_
        covars_guess = analyzer.model.covars_

        savepath = analyzer.data_dir
        plot_bootstrap_summary(results, analyzer.num_modes, save_path=savepath)

def analyze_flux_sweep_auto(data_dir, num_modes, phi_dirs, non_linear_attens, snr_threshold, init_param_dir):
    assert len(phi_dirs) == len(non_linear_attens), "Length of phi_dirs and non_linear_attens must match!"
    for phi_string, non_linear_atten in zip(phi_dirs, non_linear_attens):
        print(f"\n=== Running analysis for {phi_string} (DA={non_linear_atten}) ===")
        analyzer = HMMAnalyzer(data_dir, num_modes=num_modes)
        try:
            bootstrapping_analysis_auto(analyzer, phi_string, non_linear_atten, snr_threshold, init_param_dir)
        except Exception as e:
            print(f"Error processing {phi_string}: {e}")