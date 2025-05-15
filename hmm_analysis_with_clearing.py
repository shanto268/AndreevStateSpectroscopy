"""
HMM Analysis Module for Andreev State Spectroscopy with Clearing Tones

This module extends the base HMM analysis functionality to handle datasets
with clearing tone parameters (frequency and power).
"""

import glob
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from matplotlib.patches import Ellipse

import quasiparticleFunctions as qp
from hmm_analysis import HMMAnalyzer
from hmm_utils import create_physics_based_transition_matrix, get_means_covars


class HMMAnalyzerWithClearing(HMMAnalyzer):
    """Class for performing HMM analysis on Andreev state spectroscopy data with clearing tones."""
    
    def __init__(self, data_dir: str, num_modes: int = 3):
        """
        Initialize HMM analyzer with clearing tone support.
        
        Args:
            data_dir: Directory containing the data files
            num_modes: Number of modes to analyze
        """
        super().__init__(data_dir, num_modes)
        
        # Additional attributes for clearing tone parameters
        self.clearing_freqs = None
        self.clearing_powers = None
        self.current_clearing_freq = None
        self.current_clearing_power = None
        
    def load_data_files(self, phi_string: str) -> None:
        """
        Load data files for a specific phi value, including clearing tone parameters.
        
        Args:
            phi_string: String identifier for phi value (e.g., "phi_0p450")
        """
        data_path = os.path.join(self.data_dir, phi_string)
        self.phi = float(phi_string.split('_')[1].replace('p', '.'))
        
        # Find all clearing tone directories
        clearing_dirs = []
        for root, dirs, _ in os.walk(data_path):
            for dir_name in dirs:
                if dir_name.startswith('clearing_'):
                    clearing_dirs.append(os.path.join(root, dir_name))
        
        # Extract clearing tone parameters and data files
        self.data_files = []
        self.attenuations = []
        self.clearing_freqs = []
        self.clearing_powers = []
        
        for clearing_dir in clearing_dirs:
            # Extract frequency and power from directory name
            match = re.match(r'clearing_(\d+p\d+)GHz_(-?\d+p\d+)dBm', os.path.basename(clearing_dir))
            if match:
                freq = float(match.group(1).replace('p', '.'))
                power = float(match.group(2).replace('p', '.'))
                
                # Find binary files in this directory
                bin_files = glob.glob(os.path.join(clearing_dir, "*.bin"))
                for file in bin_files:
                    # Extract attenuation from parent directory
                    parent_dir = os.path.basename(os.path.dirname(clearing_dir))
                    atten_match = re.match(r'DA(\d+)_SR\d+', parent_dir)
                    if atten_match:
                        atten = int(atten_match.group(1))
                        self.data_files.append(file)
                        self.attenuations.append(atten)
                        self.clearing_freqs.append(freq)
                        self.clearing_powers.append(power)
        
        # Convert to numpy arrays and sort by attenuation
        self.data_files = np.array(self.data_files)
        self.attenuations = np.array(self.attenuations)
        self.clearing_freqs = np.array(self.clearing_freqs)
        self.clearing_powers = np.array(self.clearing_powers)
        
        # Sort all arrays by attenuation
        sort_idx = np.argsort(self.attenuations)[::-1]  # Descending order
        self.data_files = self.data_files[sort_idx]
        self.attenuations = self.attenuations[sort_idx]
        self.clearing_freqs = self.clearing_freqs[sort_idx]
        self.clearing_powers = self.clearing_powers[sort_idx]
    
    def find_file_index(self, target_atten: int, target_freq: Optional[float] = None, 
                       target_power: Optional[float] = None) -> int:
        """
        Find index of file with specific parameters.
        
        Args:
            target_atten: Target attenuation value
            target_freq: Target clearing frequency (optional)
            target_power: Target clearing power (optional)
            
        Returns:
            Index of matching file
        """
        mask = self.attenuations == target_atten
        if target_freq is not None:
            mask &= np.isclose(self.clearing_freqs, target_freq)
        if target_power is not None:
            mask &= np.isclose(self.clearing_powers, target_power)
        
        matches = np.where(mask)[0]
        if len(matches) == 0:
            raise ValueError(f"No files found matching the specified parameters")
        return matches[0]
    
    def load_and_process_data(self, atten: int, clearing_freq: Optional[float] = None,
                            clearing_power: Optional[float] = None, int_time: int = 2,
                            sample_rate: int = 10) -> None:
        """
        Load and process data for specific parameters.
        
        Args:
            atten: Attenuation value
            clearing_freq: Clearing frequency in GHz (optional)
            clearing_power: Clearing power in dBm (optional)
            int_time: Integration time
            sample_rate: Sample rate in MHz
        """
        self.atten = atten
        self.current_clearing_freq = clearing_freq
        self.current_clearing_power = clearing_power
        self.int_time = int_time
        
        idx = self.find_file_index(atten, clearing_freq, clearing_power)
        data_og = qp.loadAlazarData(self.data_files[idx])
        data_downsample, self.sample_rate = qp.BoxcarDownsample(
            data_og, int_time, sample_rate, returnRate=True
        )
        self.data = qp.uint16_to_mV(data_downsample)
    
    def save_analysis_results(self, states: np.ndarray, atten: int,
                            means_guess: Optional[np.ndarray] = None,
                            covars_guess: Optional[np.ndarray] = None) -> None:
        """
        Save all analysis results and plots to a results directory.
        
        Args:
            states: Decoded state sequence
            atten: Attenuation value
            means_guess: Initial means used for model (optional)
            covars_guess: Initial covariances used for model (optional)
        """
        if self.data_files is None or self.model is None:
            raise ValueError("No data or model available. Load data and fit model first.")
            
        # Get the directory of the current data file
        current_file = self.data_files[self.find_file_index(atten, 
                                                          self.current_clearing_freq,
                                                          self.current_clearing_power)]
        uid = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.path.dirname(current_file), "results", f"{uid}")
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        
        # Calculate all metrics
        mean_occ, probs = self.calculate_occupation_probabilities(states)
        snrs = self.calculate_snrs()
        rates = self.calculate_transition_rates()
        
        # Prepare results dictionary with clearing tone parameters
        results = {
            "DA": int(atten),
            "phi": float(self.phi),
            "mean_occupation": float(mean_occ),
            "probabilities": {f"P{i}": float(prob) for i, prob in enumerate(probs)},
            "snrs": {f"SNR_{i}_{j}": float(snr) 
                    for (i, j), snr in zip(self._generate_snr_pairs(), snrs)},
            "transition_rates_MHz": {f"{i}_{j}": float(rates[i, j])
                            for i in range(self.num_modes)
                            for j in range(self.num_modes)},
            "attenuation": int(atten),
            "num_modes": int(self.num_modes),
            "downSample_rate_MHz": float(self.sample_rate)
        }
        
        # Add clearing tone parameters if present
        if self.current_clearing_freq is not None:
            results["clearing_freq_GHz"] = float(self.current_clearing_freq)
        if self.current_clearing_power is not None:
            results["clearing_power_dBm"] = float(self.current_clearing_power)
        
        # Save results to JSON
        with open(os.path.join(results_dir, "analysis_results.json"), "w") as f:
            json.dump(results, f, indent=4)
            
        # Generate and save plots
        self._save_analysis_plots(results_dir, states, means_guess, covars_guess)

        # Save model
        self.save_model(f"model_{self.num_modes}_modes.pkl")
    
    def _save_analysis_plots(self, results_dir: str, states: np.ndarray,
                           means_guess: Optional[np.ndarray] = None,
                           covars_guess: Optional[np.ndarray] = None) -> None:
        """
        Save all analysis plots to the results directory.
        
        Args:
            results_dir: Directory to save plots
            states: Decoded state sequence
            means_guess: Initial means used for model (optional)
            covars_guess: Initial covariances used for model (optional)
        """
        # 1. Histogram of processed data
        plt.figure(figsize=(10, 8))
        qp.plotComplexHist(self.data[0], self.data[1], figsize=[10, 8])
        title = f"Phi: {self.phi} | DA: {self.atten} dB"
        if self.current_clearing_freq is not None:
            title += f" | Clearing: {self.current_clearing_freq} GHz"
        if self.current_clearing_power is not None:
            title += f" @ {self.current_clearing_power} dBm"
        title += f" | Integration Time: {self.int_time} μs"
        plt.title(title)
        plt.savefig(os.path.join(results_dir, "data_histogram.png"))
        plt.close()
        
        # Rest of the plotting code remains the same as in parent class
        super()._save_analysis_plots(results_dir, states, means_guess, covars_guess) 