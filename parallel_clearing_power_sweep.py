import os
from concurrent.futures import ProcessPoolExecutor

from hmm_workflows_with_clearing import clearing_power_sweep_workflow

# === USER PARAMETERS ===
# Set these before running!
base_path = "/Users/shanto/LFL/AndreevStateSpectroscopy"  # <-- Edit as needed
# clearing_data_dict should be constructed as in your notebook/code
clearing_data_dict = {}  # <-- Fill this with your actual dictionary
freqs = [9.0, 9.5, 10.0, 10.5]  # <-- List of frequencies to process
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