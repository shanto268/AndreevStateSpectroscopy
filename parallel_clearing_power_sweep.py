import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from hmm_workflows_with_clearing import clearing_power_sweep_workflow

# === USER PARAMETERS ===
# Set these before running!
base_path = r"E:\Shared drives\LFL\Projects\Quasiparticles\Andreev_Spectroscopy\051525\L1A_RUN2\phi_0p490\DA30_SR10"

# 1. Find all folders matching the convention
pattern = os.path.join(base_path, "clearing_*GHz_*dBm")
clearing_data_files = [f for f in glob.glob(pattern) if os.path.isdir(f)]

# 2. Build the dictionary
clearing_data_dict = defaultdict(list)
freq_pattern = re.compile(r"clearing_(\d+)p(\d+)GHz_")

for folder in clearing_data_files:
    match = freq_pattern.search(os.path.basename(folder))
    if match:
        freq_str = f"{match.group(1)}.{match.group(2)}"
        freq_float = float(freq_str)
        clearing_data_dict[freq_float].append(folder)

clearing_data_dict = dict(clearing_data_dict)

freqs = [9.0, 8.5, 10.0, 8]
num_modes = 2
int_time = 2
sample_rate = 10
atten = 30  # or None if not needed

# === END USER PARAMETERS ===

def run_for_freq(selected_freq):
    print(f"Starting analysis for freq: {selected_freq}")
    result = clearing_power_sweep_workflow(
        base_path,
        clearing_data_dict,
        selected_freq,
        num_modes=num_modes,
        int_time=int_time,
        sample_rate=sample_rate,
        atten=atten
    )
    print(f"Finished analysis for freq: {selected_freq}")
    return result

if __name__ == "__main__":
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Running with {num_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_for_freq, freqs))
    print("All analyses complete.") 