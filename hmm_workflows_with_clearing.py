"""
Workflow functions for HMM analysis with clearing tones.

This module provides workflow functions for analyzing Andreev state spectroscopy data
with clearing tones using Hidden Markov Models (HMM).
"""

import gc
import glob
import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import quasiparticleFunctions as qp
from hmm_analysis_with_clearing import HMMAnalyzerWithClearing
from hmm_utils import create_physics_based_transition_matrix, get_means_covars


def analyze_flux_sweep_with_clearing(
    data_dir: str,
    num_modes: int,
    non_linear_atten: int,
    clearing_freq: Optional[float] = None,
    clearing_power: Optional[float] = None,
    snr_threshold: float = 1.5
) -> None:
    """
    Analyze data across different flux values with clearing tone parameters.
    
    Args:
        data_dir: Directory containing the data files
        num_modes: Number of modes to analyze
        non_linear_atten: Starting attenuation value
        clearing_freq: Clearing frequency in GHz (optional)
        clearing_power: Clearing power in dBm (optional)
        snr_threshold: SNR threshold for stopping analysis
    """
    # Find all phi_* directories
    phi_dirs = sorted(
        [d for d in os.listdir(data_dir) if d.startswith("phi_") and os.path.isdir(os.path.join(data_dir, d))]
    )
    print(f"Found phi directories: {phi_dirs}")

    for phi_string in phi_dirs:
        print(f"\n=== Running analysis for {phi_string} ===")
        analyzer = HMMAnalyzerWithClearing(data_dir, num_modes=num_modes)
        try:
            bootstrapping_analysis_with_clearing(
                analyzer, phi_string, non_linear_atten,
                clearing_freq, clearing_power, snr_threshold
            )
        except Exception as e:
            print(f"Error processing {phi_string}: {e}")


def bootstrapping_analysis_with_clearing(
    analyzer: HMMAnalyzerWithClearing,
    phi_string: str,
    non_linear_atten: int,
    clearing_freq: Optional[float] = None,
    clearing_power: Optional[float] = None,
    snr_threshold: float = 1.5
) -> None:
    """
    Perform bootstrapping analysis with clearing tone parameters.
    
    Args:
        analyzer: HMM analyzer instance
        phi_string: String identifier for phi value
        non_linear_atten: Starting attenuation value
        clearing_freq: Clearing frequency in GHz (optional)
        clearing_power: Clearing power in dBm (optional)
        snr_threshold: SNR threshold for stopping analysis
    """
    analyzer.load_data_files(phi_string)
    attenuations = analyzer.attenuations
    idx = np.where(attenuations == non_linear_atten)[0][0]
    means_guess = None
    covars_guess = None

    results = []  # Collect results for summary plotting

    for i in range(idx, len(attenuations)):
        atten = attenuations[i]
        print(f"\n=== Processing attenuation: {atten} dB ===")
        analyzer.load_and_process_data(
            atten=atten,
            clearing_freq=clearing_freq,
            clearing_power=clearing_power
        )
        
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
        result_dict = {
            "atten": atten,
            "mean_occ": mean_occ,
            "probs": probs,
            "snrs": snrs,
            "rates": rates
        }
        
        # Add clearing tone parameters if present
        if clearing_freq is not None:
            result_dict["clearing_freq"] = clearing_freq
        if clearing_power is not None:
            result_dict["clearing_power"] = clearing_power
            
        results.append(result_dict)
        
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
    plot_bootstrap_summary_with_clearing(results, analyzer.num_modes, save_path=savepath)


def plot_bootstrap_summary_with_clearing(
    results: List[Dict],
    num_modes: int,
    save_path: Optional[str] = None
) -> None:
    """
    Plot summary of bootstrapping analysis with clearing tone parameters.
    
    Args:
        results: List of result dictionaries
        num_modes: Number of modes
        save_path: Path to save plots (optional)
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create figure directory
    if save_path:
        fig_dir = os.path.join(save_path, "Figures")
        os.makedirs(fig_dir, exist_ok=True)
    
    # Plot 1: Mean occupation vs attenuation
    plt.figure(figsize=(10, 6))
    plt.plot(df['atten'], df['mean_occ'])
    plt.title('Mean Occupation')
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('Mean State')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(os.path.join(fig_dir, "mean_occupation.png"))
    plt.close()
    
    # Plot 2: Occupation probabilities
    plt.figure(figsize=(10, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_modes))
    for i in range(num_modes):
        probs = [r['probs'][i] for r in results]
        plt.plot(df['atten'], probs, label=f'P{i}', color=colors[i])
    plt.title('Occupation Probabilities')
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(os.path.join(fig_dir, "occupation_probabilities.png"))
    plt.close()
    
    # Plot 3: SNRs
    plt.figure(figsize=(10, 6))
    snr_pairs = [(i, j) for i in range(num_modes) for j in range(i + 1, num_modes)]
    for (i, j), snrs in zip(snr_pairs, zip(*[r['snrs'] for r in results])):
        plt.plot(df['atten'], snrs, label=f'SNR_{i}_{j}')
    plt.title('Signal-to-Noise Ratios')
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('SNR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(os.path.join(fig_dir, "snrs.png"))
    plt.close()
    
    # Plot 4: Transition rates
    plt.figure(figsize=(12, 8))
    for i in range(num_modes):
        for j in range(num_modes):
            rates = [r['rates'][i, j] for r in results]
            plt.plot(df['atten'], rates, label=f'Rate_{i}_{j}')
    plt.title('Transition Rates')
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('Rate [MHz]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(os.path.join(fig_dir, "transition_rates.png"))
    plt.close()
    
    # If clearing tone parameters are present, create additional plots
    if 'clearing_freq' in df.columns and 'clearing_power' in df.columns:
        # Plot 5: Mean occupation vs clearing power for each frequency
        plt.figure(figsize=(12, 8))
        for freq in df['clearing_freq'].unique():
            mask = df['clearing_freq'] == freq
            plt.plot(df[mask]['clearing_power'], df[mask]['mean_occ'],
                    label=f'{freq} GHz', marker='o')
        plt.title('Mean Occupation vs Clearing Power')
        plt.xlabel('Clearing Power [dBm]')
        plt.ylabel('Mean State')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(os.path.join(fig_dir, "mean_occ_vs_clearing_power.png"))
        plt.close()
        
        # Plot 6: SNRs vs clearing power for each frequency
        plt.figure(figsize=(12, 8))
        for freq in df['clearing_freq'].unique():
            mask = df['clearing_freq'] == freq
            for (i, j), snrs in zip(snr_pairs, zip(*[r['snrs'] for r in results if r['clearing_freq'] == freq])):
                plt.plot(df[mask]['clearing_power'], snrs,
                        label=f'SNR_{i}_{j} @ {freq} GHz', marker='o')
        plt.title('SNRs vs Clearing Power')
        plt.xlabel('Clearing Power [dBm]')
        plt.ylabel('SNR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(os.path.join(fig_dir, "snrs_vs_clearing_power.png"))
        plt.close() 

def clearing_power_sweep_workflow(
    base_path,
    clearing_data_dict,
    selected_freq,
    num_modes=2,
    int_time=2,
    sample_rate=10,
    analyzer_class=None
):
    """
    Workflow for sweeping through clearing powers at a selected frequency.
    Args:
        base_path: root directory
        clearing_data_dict: dict mapping freq (float) to list of folder paths
        selected_freq: frequency (float) to process
        num_modes: number of HMM modes
        int_time: integration time
        sample_rate: sample rate in MHz
        analyzer_class: class to use for HMM analysis (should be HMMAnalyzer or subclass)
    """
    if analyzer_class is None:
        from hmm_analysis import HMMAnalyzer
        analyzer_class = HMMAnalyzer

    folders = clearing_data_dict.get(selected_freq, [])
    if not folders:
        print(f"No folders found for frequency {selected_freq}")
        return

    # Sort folders by power (extract from folder name)
    def extract_power(folder):
        import re
        m = re.search(r'_(\-?\d+)p(\d+)dBm', folder)
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")
        return float('inf')
    folders = sorted(folders, key=extract_power)

    prev_means = None
    prev_covars = None

    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        bin_files = glob.glob(os.path.join(folder, "*.bin"))
        if not bin_files:
            print(f"  No .bin file found in {folder}, skipping.")
            continue
        bin_file = bin_files[0]
        print(f"  Using .bin file: {bin_file}")

        # Load and process data
        data_og = qp.loadAlazarData(bin_file)
        data_downsample, sample_rate_actual = qp.BoxcarDownsample(
            data_og, int_time, sample_rate, returnRate=True
        )
        data_mV = qp.uint16_to_mV(data_downsample)
        # Memory cleanup
        del data_og
        del data_downsample

        # User input for means/covars on first file, else use previous
        if prev_means is None or prev_covars is None:
            result = get_means_covars(data_mV, num_modes)
            means_guess = result['means']
            covars_guess = result['covariances']
        else:
            means_guess = prev_means
            covars_guess = prev_covars

        # Create analyzer and run HMM
        analyzer = analyzer_class(folder, num_modes=num_modes)
        analyzer.data_files = [bin_file]
        analyzer.data = data_mV
        analyzer.sample_rate = sample_rate_actual

        analyzer.initialize_model(means_guess, covars_guess)
        analyzer.fit_model()
        logprob, states = analyzer.decode_states()
        mean_occ, probs = analyzer.calculate_occupation_probabilities(states)

        # Save model and results
        folder_name = os.path.basename(folder)
        uid = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(base_path, "Models", folder_name)
        results_dir = os.path.join(base_path, "Results", folder_name)
        figures_dir = os.path.join(base_path, "Figures", folder_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, f"model_{uid}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(analyzer.model, f)
        print(f"  Saved model to {model_path}")

        # Save results and figures
        save_full_analysis(
            analyzer, states, means_guess, covars_guess, results_dir, figures_dir, uid, atten=extract_power(folder)
        )

        # Prepare for next iteration
        prev_means = analyzer.model.means_
        prev_covars = analyzer.model.covars_

        # Memory cleanup
        del data_mV
        del analyzer
        gc.collect()

    print("\nWorkflow complete.")

def save_full_analysis(
    analyzer, states, means_guess, covars_guess, results_dir, figures_dir, uid, atten=None
):
    """
    Save all analysis results, plots, and model for a single run.
    """
    # Calculate all metrics
    mean_occ, probs = analyzer.calculate_occupation_probabilities(states)
    snrs = analyzer.calculate_snrs()
    rates = analyzer.calculate_transition_rates()
    results = {
        "DA": int(atten) if atten is not None else None,
        "mean_occupation": float(mean_occ),
        "probabilities": {f"P{i}": float(prob) for i, prob in enumerate(probs)},
        "snrs": {f"SNR_{i}_{j}": float(snr)
                 for (i, j), snr in zip(analyzer._generate_snr_pairs(), snrs)},
        "transition_rates_MHz": {f"{i}_{j}": float(rates[i, j])
                                for i in range(analyzer.num_modes)
                                for j in range(analyzer.num_modes)},
        "attenuation": int(atten) if atten is not None else None,
        "num_modes": int(analyzer.num_modes),
        "downSample_rate_MHz": float(analyzer.sample_rate)
    }
    # Save results to JSON
    with open(os.path.join(results_dir, f"analysis_results_{uid}.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save figures
    # 1. Histogram of processed data
    plt.figure(figsize=(10, 8))
    qp.plotComplexHist(analyzer.data[0], analyzer.data[1], figsize=[10, 8])
    plt.title(f"DA: {atten} dB | Integration Time: {getattr(analyzer, 'int_time', '?')} μs")
    plt.savefig(os.path.join(figures_dir, f"data_histogram_{uid}.png"))
    plt.close()

    # 2. Data with initial means and covariances
    if means_guess is not None and covars_guess is not None:
        plt.figure(figsize=(10, 8))
        ax = qp.plotComplexHist(analyzer.data[0], analyzer.data[1], figsize=[10, 8])
        analyzer._plot_means_and_covars(ax, means_guess, covars_guess, "Initial")
        plt.savefig(os.path.join(figures_dir, f"data_with_initial_parameters_{uid}.png"))
        plt.close()

    # 3. Data with trained means and covariances
    plt.figure(figsize=(10, 8))
    ax = qp.plotComplexHist(analyzer.data[0], analyzer.data[1], figsize=[10, 8])
    analyzer._plot_means_and_covars(ax, analyzer.model.means_, analyzer.model.covars_, "Trained")
    plt.savefig(os.path.join(figures_dir, f"data_with_trained_parameters_{uid}.png"))
    plt.close()

    # 4. Timeseries slices (optional, as in your code)
    change_indices = np.where(np.diff(states) != 0)[0] + 1
    if len(change_indices) >= 2:
        i = np.random.choice(len(change_indices) - 1)
        j = i + np.random.randint(1, 6)
        try:
            analyzer._plot_timeseries_slice(states, change_indices[i]-5, change_indices[i]+5,
                                      f"short_transition_{uid}", figures_dir)
        except Exception as e:
            analyzer._plot_timeseries_slice(states, change_indices[i], change_indices[i]+5,
                                      f"short_transition_{uid}", figures_dir)
        if i + 1 < len(change_indices):
            analyzer._plot_timeseries_slice(states, change_indices[i], change_indices[j],
                                      f"between_transitions_{uid}", figures_dir)