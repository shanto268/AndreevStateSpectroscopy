import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# === USER PARAMETERS ===
base_path = r"E:\Shared drives\LFL\Projects\Quasiparticles\Andreev_Spectroscopy\051525\L1A_RUN2\phi_0p490\DA30_SR10\Results"  # <-- Edit as needed
report_path = r"E:\Shared drives\LFL\Projects\Quasiparticles\Andreev_Spectroscopy\051525\L1A_RUN2\phi_0p490"  # <-- Edit as needed


# 1. Find all clearing_*GHz_*dBm folders
pattern = os.path.join(base_path, "clearing_*GHz_*dBm")
folders = glob.glob(pattern)

records = []
for folder in folders:
    # Extract freq and power from folder name
    m = re.search(r'clearing_(\d+p\d+)GHz_(-?\d+p\d+)dBm', os.path.basename(folder))
    if not m:
        continue
    freq = float(m.group(1).replace('p', '.'))
    power = float(m.group(2).replace('p', '.'))
    # Find the json file
    json_files = glob.glob(os.path.join(folder, "analysis_results_*.json"))
    if not json_files:
        continue
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    # Flatten the record
    record = {
        'freq': freq,
        'power': power,
        **data
    }
    records.append(record)

# Create DataFrame
if not records:
    raise RuntimeError("No records found!")
df = pd.DataFrame(records)

# Ensure report_path exists
os.makedirs(report_path, exist_ok=True)
pdf_file = os.path.join(report_path, "clearing_tone_analysis_report.pdf")

with PdfPages(pdf_file) as pdf:
    # 1. Mean occupation as a function of clearing tone frequency and power
    plt.figure(figsize=(10, 6))
    scatter1 = sns.scatterplot(data=df, x='freq', y='power', hue='mean_occupation', palette='viridis', s=100)
    plt.title('Mean Occupation vs Frequency and Power')
    plt.xlabel('Clearing Tone Frequency (GHz)')
    plt.ylabel('Clearing Tone Power (dBm)')
    plt.colorbar(scatter1.collections[0], label='Mean Occupation')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 2. Release rates as a function of clearing tone frequency and power
    df['release_rate'] = df['transition_rates_MHz'].apply(lambda d: d.get('1_0') if isinstance(d, dict) else None)
    plt.figure(figsize=(10, 6))
    scatter2 = sns.scatterplot(data=df, x='freq', y='power', hue='release_rate', palette='magma', s=100)
    plt.title('Release Rate (1→0) vs Frequency and Power')
    plt.xlabel('Clearing Tone Frequency (GHz)')
    plt.ylabel('Clearing Tone Power (dBm)')
    plt.colorbar(scatter2.collections[0], label='Release Rate (MHz)')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 3. 4-axis plot: mean occupation, release rates, fitted release, SNRs vs power for all frequencies
    freqs = sorted(df['freq'].unique())
    for freq in freqs:
        sub = df[df['freq'] == freq].sort_values('power')
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color1 = 'tab:blue'
        ax1.set_xlabel('Clearing Tone Power (dBm)')
        ax1.set_ylabel('Mean Occupation', color=color1)
        ax1.plot(sub['power'], sub['mean_occupation'], 'o-', color=color1, label='Mean Occ')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Release Rate (MHz)', color=color2)
        ax2.plot(sub['power'], sub['release_rate'], 's-', color=color2, label='Release Rate')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # SNR: use SNR_0_1
        if 'snrs' in sub.columns and sub['snrs'].iloc[0] and 'SNR_0_1' in sub['snrs'].iloc[0]:
            ax3 = ax1.twinx()
            color3 = 'tab:green'
            ax3.spines['right'].set_position(('outward', 60))
            ax3.set_ylabel('SNR (0_1)', color=color3)
            ax3.plot(sub['power'], [snr.get('SNR_0_1') if isinstance(snr, dict) else None for snr in sub['snrs']], '^-', color=color3, label='SNR_0_1')
            ax3.tick_params(axis='y', labelcolor=color3)
        plt.title(f'Freq {freq} GHz: Mean Occ, Release Rate, SNR vs Power')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def dbm_to_watts(dbm):
        """Converts power from dBm to watts."""
        return 10 ** ((dbm - 30) / 10)

    # Power law fit plots
    for freq in freqs:
        sub = df[df['freq'] == freq].sort_values('power')
        mask = sub['release_rate'].notnull()
        powers = sub['power'][mask].values
        rates = sub['release_rate'][mask].values
        if len(powers) < 2:
            continue
        watts = np.array([dbm_to_watts(p) for p in powers])
        log_power = np.log10(watts)
        log_rate = np.log10(rates)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(log_power, log_rate, 'o-', label='Data')
        ax1.set_xlabel('log10(Power [W])')
        ax1.set_ylabel('log10(Clearing Rate [MHz])')
        ax1.set_title(f'Power Law Fit: Freq {freq} GHz')
        coeffs = np.polyfit(log_power, log_rate, 1)
        fit_line = np.polyval(coeffs, log_power)
        ax1.plot(log_power, fit_line, '--', label=f'Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')
        ax1.legend()
        pdf.savefig(fig)
        plt.close(fig)
        print(f"Freq {freq} GHz: log10(rate) = {coeffs[0]:.3f} * log10(power) + {coeffs[1]:.3f}")

print(f"PDF report saved to: {pdf_file}")
