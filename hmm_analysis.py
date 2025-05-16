"""
HMM Analysis Module for Andreev State Spectroscopy

This module provides functionality for analyzing Andreev state spectroscopy data
using Hidden Markov Models (HMM).
"""

import gc  # Add this import at the top of your file
import glob
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse

import quasiparticleFunctions as qp
from hmm_utils import create_physics_based_transition_matrix, get_means_covars


class HMMAnalyzer:
    """Class for performing HMM analysis on Andreev state spectroscopy data."""
    
    def __init__(self, data_dir: str, num_modes: int = 3):
        """
        Initialize HMM analyzer.
        
        Args:
            data_dir: Directory containing the data files
            num_modes: Number of modes to analyze
        """
        self.data_dir = data_dir
        self.num_modes = num_modes
        self.figure_path = os.path.join(data_dir, "Figures")
        os.makedirs(self.figure_path, exist_ok=True)
        
        # Initialize attributes
        self.data_files = None
        self.attenuations = None
        self.model = None
        self.data = None
        self.sample_rate = None
        self.results_dir = None
        self.phi = None
        self.atten = None

    def load_data_files(self, phi_string: str) -> None:
        """
        Load data files for a specific phi value.
        
        Args:
            phi_string: String identifier for phi value (e.g., "phi_0p450")
        """
        data_path = os.path.join(self.data_dir, phi_string)
        self.phi = float(phi_string.split('_')[1].replace('p', '.'))
        self.data_files = glob.glob(os.path.join(data_path, "**", "*.bin"), recursive=True)
        
        # Extract and sort attenuations
        self.attenuations = []
        for file in self.data_files:
            _, s = file.split('DA')
            atten = s[:2]
            self.attenuations.append(int(atten))
            
        sort_idx = np.argsort(self.attenuations)
        self.data_files = np.asarray(self.data_files)[sort_idx[::-1]]
        self.attenuations = np.asarray(self.attenuations)[sort_idx[::-1]]
        
    def find_atten_file_index(self, target_atten: int) -> int:
        """Find index of file with specific attenuation."""
        return np.where(self.attenuations == target_atten)[0][0]
    
    def load_and_process_data(self, atten: int, int_time: int = 2, 
                            sample_rate: int = 10) -> None:
        """
        Load and process data for specific attenuation.
        
        Args:
            atten: Attenuation value
            int_time: Integration time
            sample_rate: Sample rate in MHz
        """
        self.atten = atten
        self.int_time = int_time
        idx = self.find_atten_file_index(atten)
        data_og = qp.loadAlazarData(self.data_files[idx])
        data_downsample, self.sample_rate = qp.BoxcarDownsample(
            data_og, int_time, sample_rate, returnRate=True
        )
        self.data = qp.uint16_to_mV(data_downsample)
        
    def initialize_model(self, means_guess: Optional[np.ndarray] = None,
                        covars_guess: Optional[np.ndarray] = None) -> None:
        """
        Initialize HMM model with optional initial parameters.
        
        Args:
            means_guess: Initial means for the model
            covars_guess: Initial covariances for the model
        """
        self.model = hmm.GaussianHMM(
            n_components=self.num_modes,
            covariance_type='full',
            init_params='s',
            n_iter=500,
            tol=0.001,
            verbose=True
        )
        
        if means_guess is not None:
            self.model.means_ = means_guess
        if covars_guess is not None:
            self.model.covars_ = covars_guess
            
        self.model.transmat_ = create_physics_based_transition_matrix(self.num_modes)
        
    def fit_model(self) -> None:
        """Fit the HMM model to the data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_and_process_data first.")
        self.model.fit(self.data.T)
        
    def calculate_snrs(self) -> np.ndarray:
        """Calculate SNRs between all mode pairs."""
        snr_pairs = self._generate_snr_pairs()
        return np.array([qp.getSNRhmm(self.model, mode1=pair[0], mode2=pair[1]) 
                        for pair in snr_pairs])
    
    def _generate_snr_pairs(self) -> List[Tuple[int, int]]:
        """Generate all valid SNR pairs for the current number of modes."""
        return [(i, j) for i in range(self.num_modes) 
                for j in range(i + 1, self.num_modes)]
    
    def decode_states(self) -> Tuple[float, np.ndarray]:
        """Decode the hidden states using Viterbi algorithm."""
        return self.model.decode(self.data.T, algorithm='viterbi')
    
    def calculate_occupation_probabilities(self, states: np.ndarray) -> Tuple[float, List[float]]:
        """
        Calculate occupation probabilities for each mode.
        
        Returns:
            Tuple of (mean occupation, list of probabilities)
        """
        mean_occupation = np.mean(states)
        probabilities = [np.sum(states == i)/states.size for i in range(self.num_modes)]
        return mean_occupation, probabilities
    
    def calculate_transition_rates(self) -> np.ndarray:
        """
        Calculate transition rates from the model's transition matrix.
        
        Returns:
            Array of transition rates in MHz
        """
        if self.model is None:
            raise ValueError("No model available. Fit the model first.")
        return qp.getTransRatesFromProb(self.sample_rate, self.model.transmat_)
    
    def print_transition_rates(self) -> None:
        """Print transition rates in a readable format."""
        rates = self.calculate_transition_rates()
        print("Transition rates [MHz]:")
        for i in range(rates.shape[0]):
            for j in range(rates.shape[1]):
                print(f"{i} → {j}: {rates[i,j]:.6f} MHz")
    
    def analyze_lifetimes(self, states: np.ndarray, time: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Analyze lifetimes for each state.
        
        Args:
            states: Decoded state sequence
            time: Time array corresponding to states
            
        Returns:
            Dictionary mapping state numbers to arrays of lifetimes
        """
        return qp.extractLifetimes(states, time)
    
    def analyze_anti_lifetimes(self, states: np.ndarray, time: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Analyze anti-lifetimes (time between state changes) for each state.
        
        Args:
            states: Decoded state sequence
            time: Time array corresponding to states
            
        Returns:
            Dictionary mapping state numbers to arrays of anti-lifetimes
        """
        return qp.extractAntiLifetimes(states, time)
    
    def plot_states(self, states: np.ndarray, time_window: Tuple[int, int], 
                   plot_name: str = "states") -> None:
        """
        Plot and save the decoded states and raw data.
        
        Args:
            states: Decoded state sequence
            time_window: Tuple of (start_index, end_index) for plotting
            plot_name: Name for the saved plot file
        """
        time = np.arange(len(states))/self.sample_rate
        sind, eind = time_window
        
        fig, ax = plt.subplots(2, 1, figsize=[9, 6])
        ax[0].plot(time[sind:eind], states[sind:eind], label='HMM')
        ax[0].set_title(f"Average Occupation: {states.mean()}")
        ax[0].legend()
        ax[0].set_ylabel('Occupation')
        
        ax[1].set_ylabel('mV')
        ax[1].set_xlabel('Time [μs]')
        ax[1].plot(time[sind:eind], self.data[0, sind:eind], label='real')
        ax[1].plot(time[sind:eind], self.data[1, sind:eind], label='imag')
        ax[1].legend()
        
        plt.savefig(os.path.join(self.results_dir, f"{plot_name}.png"))
        plt.close()
        
    def plot_complex_histogram(self, plot_name: str = "complex_histogram") -> None:
        """
        Plot and save complex histogram with means and covariance ellipses.
        
        Args:
            plot_name: Name for the saved plot file
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = qp.plotComplexHist(self.data[0], self.data[1], figsize=[14, 14])
        
        # Calculate dynamic limits with some padding
        x_min = min(self.data[0].min(), self.model.means_[:, 0].min())
        x_max = max(self.data[0].max(), self.model.means_[:, 0].max())
        y_min = min(self.data[1].min(), self.model.means_[:, 1].min())
        y_max = max(self.data[1].max(), self.model.means_[:, 1].max())
        
        # Add 10% padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        for i in range(len(self.model.means_)):
            # Plot mean
            ax.scatter(self.model.means_[i, 0], self.model.means_[i, 1],
                      c=f'C{i}', s=200, marker='x', linewidth=3)
            
            # Add text label
            ax.text(self.model.means_[i, 0], self.model.means_[i, 1],
                   f' {i}', fontsize=14, color=f'C{i}', fontweight='bold')
            
            # Plot covariance ellipse
            eigenvals, eigenvecs = np.linalg.eigh(self.model.covars_[i])
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 4 * np.sqrt(eigenvals)
            ellip = Ellipse(self.model.means_[i], width, height, angle,
                          facecolor='none', edgecolor=f'C{i}',
                          alpha=0.8, linestyle='--')
            plt.gca().add_patch(ellip)
            
        plt.xlabel("I [mV]")
        plt.ylabel("Q [mV]")
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal')
        plt.savefig(os.path.join(self.results_dir, f"{plot_name}.png"))
        plt.close()
        
    def plot_lifetime_histogram(self, lifetimes: Dict[int, np.ndarray], 
                              state: int, atten: int, bins: int = 200) -> None:
        """
        Plot and save lifetime histogram for a specific state.
        
        Args:
            lifetimes: Dictionary of lifetimes from analyze_lifetimes
            state: State number to plot
            atten: Attenuation value for title
            bins: Number of histogram bins
        """
        if state in lifetimes and len(lifetimes[state]):
            h = qp.fitAndPlotExpDecay(lifetimes[state], 
                                     cut=np.mean(lifetimes[state]), 
                                     bins=bins)
            plt.title(f'LT mode {state} | DA {atten} dB')
            plt.savefig(os.path.join(self.results_dir, f"lifetime_state_{state}.png"))
            plt.close()
    
    def plot_anti_lifetime_histogram(self, anti_lifetimes: Dict[int, np.ndarray], 
                                   state: int, atten: int, bins: int = 200) -> None:
        """
        Plot and save anti-lifetime histogram for a specific state.
        
        Args:
            anti_lifetimes: Dictionary of anti-lifetimes from analyze_anti_lifetimes
            state: State number to plot
            atten: Attenuation value for title
            bins: Number of histogram bins
        """
        if state in anti_lifetimes and len(anti_lifetimes[state]):
            h = qp.fitAndPlotExpDecay(anti_lifetimes[state], 
                                     cut=np.mean(anti_lifetimes[state]), 
                                     bins=bins)
            plt.title(f'antiLT mode {state} | DA {atten} dB')
            plt.savefig(os.path.join(self.results_dir, f"anti_lifetime_state_{state}.png"))
            plt.close()
    
    def analyze_all_lifetimes(self, states: np.ndarray, atten: int) -> None:
        """
        Analyze and plot all lifetimes and anti-lifetimes.
        
        Args:
            states: Decoded state sequence
            atten: Attenuation value for plot titles
        """
        time = np.arange(len(states))/self.sample_rate
        
        # Analyze lifetimes
        lifetimes = self.analyze_lifetimes(states, time)
        for state in range(self.num_modes):
            self.plot_lifetime_histogram(lifetimes, state, atten)
        
        # Analyze anti-lifetimes
        anti_lifetimes = self.analyze_anti_lifetimes(states, time)
        for state in range(self.num_modes):
            self.plot_anti_lifetime_histogram(anti_lifetimes, state, atten)
    
    def save_model(self, model_name: str) -> None:
        """Save the trained model to a file."""
        model_path = f"{self.results_dir}/{model_name}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
            
    @classmethod
    def load_model(cls, model_path: str) -> 'HMMAnalyzer':
        """Load a trained model from a file."""
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        analyzer = cls("")  # Empty data_dir as it's not needed for loaded model
        analyzer.model = model
        return analyzer

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
            
        uid = datetime.now().strftime("%Y%m%d_%H%M%S")
        if atten is not None and self.attenuations is not None:
            current_file = self.data_files[self.find_atten_file_index(atten)]
            results_dir = os.path.join(os.path.dirname(current_file), "results", f"{uid}")
        else:
            current_file = self.data_files[0]
            results_dir = os.path.join(os.path.dirname(current_file), "results", f"{uid}")
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        
        # Calculate all metrics
        mean_occ, probs = self.calculate_occupation_probabilities(states)
        snrs = self.calculate_snrs()
        rates = self.calculate_transition_rates()
        
        # Prepare results dictionary
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
        plt.title(f"Phi: {self.phi} | DA: {self.atten} dB | Integration Time: {self.int_time} μs")
        plt.savefig(os.path.join(results_dir, "data_histogram.png"))
        plt.close()
        
        # 2. Data with initial means and covariances
        if means_guess is not None and covars_guess is not None:
            plt.figure(figsize=(10, 8))
            ax = qp.plotComplexHist(self.data[0], self.data[1], figsize=[10, 8])
            self._plot_means_and_covars(ax, means_guess, covars_guess, "Initial")
            plt.savefig(os.path.join(results_dir, "data_with_initial_parameters.png"))
            plt.close()
        
        # 3. Data with trained means and covariances
        plt.figure(figsize=(10, 8))
        ax = qp.plotComplexHist(self.data[0], self.data[1], figsize=[10, 8])
        self._plot_means_and_covars(ax, self.model.means_, self.model.covars_, "Trained")
        plt.savefig(os.path.join(results_dir, "data_with_trained_parameters.png"))
        plt.close()
        
        # 4. Timeseries slices
        change_indices = np.where(np.diff(states) != 0)[0] + 1
        if len(change_indices) >= 2:
            i = np.random.choice(len(change_indices) - 1)
            j = i + np.random.randint(1, 6)
            
            # Short slice around a transition
            try:
                self._plot_timeseries_slice(states, change_indices[i]-5, change_indices[i]+5,
                                      "short_transition", results_dir)
            except Exception as e:
                self._plot_timeseries_slice(states, change_indices[i], change_indices[i]+5,
                                      "short_transition", results_dir)
                print(f"Error plotting short transition: {e}")
            
            # Longer slice between transitions
            if i + 1 < len(change_indices):
                self._plot_timeseries_slice(states, change_indices[i], change_indices[j],
                                          "between_transitions", results_dir)
    
    def _plot_means_and_covars(self, ax: plt.Axes, means: np.ndarray, 
                              covars: np.ndarray, prefix: str) -> None:
        """Helper method to plot means and covariances on an axis."""
        for i in range(len(means)):
            # Plot mean
            ax.scatter(means[i, 0], means[i, 1],
                      c=f'C{i}', s=200, marker='x', linewidth=3)
            # Add text label
            ax.text(means[i, 0], means[i, 1],
                   f' {i}', fontsize=14, color=f'C{i}', fontweight='bold')
            # Plot covariance ellipse
            eigenvals, eigenvecs = np.linalg.eigh(covars[i])
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 4 * np.sqrt(eigenvals)
            ellip = Ellipse(means[i], width, height, angle,
                          facecolor='none', edgecolor=f'C{i}',
                          alpha=0.8, linestyle='--')
            plt.gca().add_patch(ellip)
        plt.title(f"{prefix} Means and Covariances")
        plt.xlabel("I [mV]")
        plt.ylabel("Q [mV]")
        # Dynamic axis limits with padding
        x_min = min(self.data[0].min(), means[:, 0].min())
        x_max = max(self.data[0].max(), means[:, 0].max())
        y_min = min(self.data[1].min(), means[:, 1].min())
        y_max = max(self.data[1].max(), means[:, 1].max())
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal')

    def _plot_timeseries_slice(self, states: np.ndarray, start_idx: int, 
                              end_idx: int, name: str, results_dir: str) -> None:
        """Helper method to plot a slice of the timeseries."""
        # Clamp indices to valid range
        n = len(states)
        start_idx = max(0, min(start_idx, n-1))
        end_idx = max(start_idx+1, min(end_idx, n))
        time = np.arange(n)/self.sample_rate
        # Calculate dynamic limits with padding
        y_min = min(self.data[0, start_idx:end_idx].min(), 
                   self.data[1, start_idx:end_idx].min())
        y_max = max(self.data[0, start_idx:end_idx].max(), 
                   self.data[1, start_idx:end_idx].max())
        y_padding = (y_max - y_min) * 0.1
        fig, ax = plt.subplots(2, 1, figsize=[9, 6])
        ax[0].plot(time[start_idx:end_idx], states[start_idx:end_idx], label='HMM')
        ax[0].set_title(f"Average Occupation: {states.mean():.3f}")
        ax[0].legend()
        ax[0].set_ylabel('Occupation')
        ax[1].set_ylabel('mV')
        ax[1].set_xlabel('Time [μs]')
        ax[1].plot(time[start_idx:end_idx], self.data[0, start_idx:end_idx], label='real')
        ax[1].plot(time[start_idx:end_idx], self.data[1, start_idx:end_idx], label='imag')
        ax[1].legend()
        ax[1].set_ylim(y_min - y_padding, y_max + y_padding)
        plt.savefig(os.path.join(results_dir, f"timeseries_{name}.png"))
        plt.close()

class RefHMMAnalyzer(HMMAnalyzer):
    def process_ref_folders(self, ref_folders, num_modes=2, int_time=2, sample_rate=10):
        """
        Process all .bin files in the given reference folders using the standard single-power workflow.
        """
        results = []
        for folder in ref_folders:
            bin_files = glob.glob(os.path.join(folder, "*.bin"))
            if not bin_files:
                print(f"No .bin file found in {folder}")
                continue
            bin_file = bin_files[0]
            print(f"Processing {bin_file}")

            # Set up analyzer for this file
            self.data_files = [bin_file]
            self.attenuations = [None]
            self.data_dir = folder
            self.figure_path = os.path.join(folder, "Figures")
            os.makedirs(self.figure_path, exist_ok=True)

            # Load and process data
            data_og = qp.loadAlazarData(bin_file)
            data_downsample, self.sample_rate = qp.BoxcarDownsample(
                data_og, int_time, sample_rate, returnRate=True
            )
            self.data = qp.uint16_to_mV(data_downsample)

            # Get initial parameters
            result = get_means_covars(self.data, num_modes)
            means_guess = result['means']
            covars_guess = result['covariances']

            # Initialize and fit model
            self.initialize_model(means_guess, covars_guess)
            self.fit_model()

            # Decode states and calculate probabilities
            logprob, states = self.decode_states()
            mean_occ, probs = self.calculate_occupation_probabilities(states)
            print(f"Mean occupation: {mean_occ}")
            print(f"Probabilities: {probs}")

            # Save all results and plots
            self.save_analysis_results(states, None, means_guess, covars_guess)

            # Calculate SNRs
            snrs = self.calculate_snrs()
            print("SNRs:", snrs)

            results.append({
                "bin_file": bin_file,
                "mean_occ": mean_occ,
                "probs": probs,
                "snrs": snrs
            })

            # Memory cleanup
            del data_og
            del data_downsample
            del self.data
            gc.collect()
        return results

    def plot_ref_files_to_pdf(self, ref_files, int_time=2, sample_rate=10):
        """
        Args:
            ref_files: list of folder paths, each containing a .bin file
            int_time: integration time in microseconds
            sample_rate: sample rate in MHz
        For each folder in ref_files, load the .bin file, plot a 2D I-Q histogram, and save all plots to a single PDF.
        The plot title will be the folder name (e.g., no_clearing_REF_for_10p00GHz).
        The PDF will be saved in .../Figures/RefFiles with a unique timestamped filename.
        """
        # Set up save directory
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(ref_files[0])),  # up two levels from a ref file
            "Figures", "RefFiles"
        )
        os.makedirs(save_dir, exist_ok=True)
        # Unique filename
        uid = datetime.now().strftime("%H%M%S-%m%d%Y")
        pdf_path = os.path.join(save_dir, f"ref_files_{uid}.pdf")

        with PdfPages(pdf_path) as pdf:
            for folder in ref_files:
                bin_files = glob.glob(os.path.join(folder, "*.bin"))
                if not bin_files:
                    print(f"No .bin file found in {folder}")
                    continue
                bin_file = bin_files[0]
                # Load data
                data_og = qp.loadAlazarData(bin_file)
                data_downsample, self.sample_rate = qp.BoxcarDownsample(
                        data_og, int_time, sample_rate, returnRate=True
                )
                self.data = qp.uint16_to_mV(data_downsample)
                # 2D I-Q histogram plot
                plt.figure(figsize=(12, 10))
                h = plt.hist2d(self.data[0], self.data[1], bins=80, 
                               norm=plt.matplotlib.colors.LogNorm(), cmap='Greys')
                plt.colorbar(h[3], shrink=0.9, extend='both')
                plt.title(os.path.basename(folder))
                plt.xlabel("I [mV]")
                plt.ylabel("Q [mV]")
                plt.grid(True, alpha=0.3)
                plt.gca().set_aspect('equal')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                # Explicitly delete large arrays and collect garbage
                del data_og
                del data_downsample
                del self.data
                gc.collect()
        print(f"Saved all reference plots to {pdf_path}")

    def process_ref_file_by_index(self, ref_files, index, num_modes=2, int_time=2, sample_rate=10):
        """
        Process a single .bin file in ref_files by index using the standard single-power workflow.
        Args:
            ref_files: list of folder paths, each containing a .bin file
            index: integer index into ref_files
            num_modes: number of HMM modes
            int_time: integration time
            sample_rate: sample rate in MHz
        """
        if not (0 <= index < len(ref_files)):
            print(f"Invalid index {index}. Allowed values: 0 to {len(ref_files)-1}")
            print("Available files:")
            for i, folder in enumerate(ref_files):
                print(f"  [{i}]: {folder}")
            return
        folder = ref_files[index]
        print(f"Processing index {index}: {folder}")
        bin_files = glob.glob(os.path.join(folder, "*.bin"))
        if not bin_files:
            print(f"No .bin file found in {folder}")
            return
        bin_file = bin_files[0]
        print(f"  Using .bin file: {bin_file}")
        # Set up analyzer for this file
        self.data_files = [bin_file]
        self.attenuations = [None]
        self.data_dir = folder
        self.figure_path = os.path.join(folder, "Figures")
        os.makedirs(self.figure_path, exist_ok=True)
        # Load and process data
        data_og = qp.loadAlazarData(bin_file)
        data_downsample, self.sample_rate = qp.BoxcarDownsample(
            data_og, int_time, sample_rate, returnRate=True
        )
        self.data = qp.uint16_to_mV(data_downsample)
        # Get initial parameters
        result = get_means_covars(self.data, num_modes)
        means_guess = result['means']
        covars_guess = result['covariances']
        # Initialize and fit model
        self.initialize_model(means_guess, covars_guess)
        self.fit_model()
        # Decode states and calculate probabilities
        logprob, states = self.decode_states()
        mean_occ, probs = self.calculate_occupation_probabilities(states)
        print(f"Mean occupation: {mean_occ}")
        print(f"Probabilities: {probs}")
        # Save all results and plots in RefFiles directory
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(folder)),
            "Figures", "RefFiles"
        )
        os.makedirs(save_dir, exist_ok=True)
        self.results_dir = save_dir
        self.save_analysis_results(states, None, means_guess, covars_guess)
        # Calculate SNRs
        snrs = self.calculate_snrs()
        print("SNRs:", snrs)
        # Memory cleanup
        del data_og
        del data_downsample
        del self.data
        gc.collect()

    def plot_all_ref_means(self, ref_files, int_time=2, sample_rate=10, point_size=100, cmap='viridis'):
        """
        Create a single plot showing the mean I,Q values for each reference file, 
        colored by the frequency value extracted from the folder name.
        
        Args:
            ref_files: list of folder paths, each containing a .bin file
            int_time: integration time in microseconds
            sample_rate: sample rate in MHz
            point_size: size of the scatter points
            cmap: colormap to use for frequency values
        
        The plot will be saved in .../Figures/RefFiles with a unique timestamped filename.
        Folder names are expected to have the format "*_for_XXpYYGHz" where XXpYY is the frequency.
        """
        # Set up save directory
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(ref_files[0])),  # up two levels from a ref file
            "Figures", "RefFiles"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # Unique filename
        uid = datetime.now().strftime("%H%M%S-%m%d%Y")
        png_path = os.path.join(save_dir, f"ref_means_map_{uid}.png")
        
        # Prepare the figure
        plt.figure(figsize=(14, 12))
        
        # Arrays to store data for colorbar
        all_means_I = []
        all_means_Q = []
        all_freqs = []
        all_labels = []
        
        # Process each folder
        for folder in ref_files:
            bin_files = glob.glob(os.path.join(folder, "*.bin"))
            if not bin_files:
                print(f"No .bin file found in {folder}")
                continue
            
            # Extract frequency from folder name
            try:
                folder_basename = os.path.basename(folder)
                if "for_" in folder_basename and "GHz" in folder_basename:
                    freq_str = folder_basename.split("for_")[1].split("GHz")[0]
                    frequency = float(freq_str.replace("p", "."))
                else:
                    print(f"Could not extract frequency from folder name: {folder_basename}")
                    frequency = len(all_freqs)  # Use index as fallback
                
                # Load and process data
                bin_file = bin_files[0]
                data_og = qp.loadAlazarData(bin_file)
                data_downsample, _ = qp.BoxcarDownsample(
                    data_og, int_time, sample_rate, returnRate=True
                )
                data = qp.uint16_to_mV(data_downsample)
                
                # Calculate mean I and Q values
                mean_I = np.mean(data[0])
                mean_Q = np.mean(data[1])
                
                # Store values for plotting
                all_means_I.append(mean_I)
                all_means_Q.append(mean_Q)
                all_freqs.append(frequency)
                all_labels.append(folder_basename)
                
                # Free memory
                del data_og
                del data_downsample
                del data
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {folder}: {e}")
        
        # Convert to numpy arrays for plotting
        all_means_I = np.array(all_means_I)
        all_means_Q = np.array(all_means_Q)
        all_freqs = np.array(all_freqs)
        
        # Create scatter plot with frequency-based coloring
        scatter = plt.scatter(all_means_I, all_means_Q, 
                            c=all_freqs, 
                            cmap=cmap, 
                            s=point_size, 
                            alpha=0.8,
                            edgecolors='k')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, shrink=0.85)
        cbar.set_label('Frequency (GHz)', fontsize=12)
        
        # Add labels for each point
        for i, label in enumerate(all_labels):
            plt.annotate(f"{all_freqs[i]:.2f} GHz", 
                        (all_means_I[i], all_means_Q[i]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9)
        
        # Plot formatting
        plt.xlabel("I [mV]", fontsize=14)
        plt.ylabel("Q [mV]", fontsize=14)
        plt.title("Reference File Means by Frequency", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal')
        
        # Add some padding to the plot
        x_range = np.max(all_means_I) - np.min(all_means_I)
        y_range = np.max(all_means_Q) - np.min(all_means_Q)
        padding = 0.1 * max(x_range, y_range)
        plt.xlim(np.min(all_means_I) - padding, np.max(all_means_I) + padding)
        plt.ylim(np.min(all_means_Q) - padding, np.max(all_means_Q) + padding)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
        
        print(f"Saved reference means map to {png_path}")
        
        # Return the path to the saved file
        return png_path

def main():
    """Example usage of the HMMAnalyzer class."""
    # Initialize analyzer
    data_dir = "path/to/your/data"
    analyzer = HMMAnalyzer(data_dir, num_modes=3)
    
    # Load data files
    analyzer.load_data_files("phi_0p450")
    
    # Process data for specific attenuation
    atten = 26
    analyzer.load_and_process_data(atten=atten)
    
    # Get initial parameters
    result = get_means_covars(analyzer.data, analyzer.num_modes)
    means_guess = result['means']
    covars_guess = result['covariances']
    
    # Initialize and fit model
    analyzer.initialize_model(means_guess, covars_guess)
    analyzer.fit_model()
    
    # Calculate SNRs
    snrs = analyzer.calculate_snrs()
    print("SNRs:", snrs)
    
    # Decode states and calculate probabilities
    logprob, states = analyzer.decode_states()
    mean_occ, probs = analyzer.calculate_occupation_probabilities(states)
    
    # Calculate and print transition rates
    analyzer.print_transition_rates()
    
    # Analyze lifetimes
    analyzer.analyze_all_lifetimes(states, atten)
    
    # Save all results and plots
    analyzer.save_analysis_results(states, atten, means_guess, covars_guess)
    
    # Save model
    analyzer.save_model("path/to/save/model.pkl")

if __name__ == "__main__":
    main()