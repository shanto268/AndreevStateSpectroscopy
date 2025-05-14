# Andreev State Spectroscopy

This repository contains the code for the Andreev State Spectroscopy project.

## Data Structure

The experimental data is organized in the following hierarchical structure:

```
Quasiparticles/
├── Andreev_Spectroscopy/
│   └── {DATE}/                    # Format: MMDDYY
│       └── {RUN_NAME}/            # e.g., L1A_RUN1
│           ├── phi_{PHI_VALUE}/   # Format: phi_0p300, phi_0p322, etc.
│           │   └── DA{DA}_SR{SR}/ # Format: DA16_SR10, DA18_SR10, etc.
│           │       └── clearing_{FREQ}GHz_{POWER}dBm/
│           │           ├── {RUN_NAME}_{TIMESTAMP}.bin  # Binary data file
│           │           └── {RUN_NAME}_{TIMESTAMP}.txt  # Metadata file
│           ├── Figures/           # Generated figures
│           ├── vna_fit_{PHI}.pkl  # VNA fit data for each phi value
│           └── MEASUREMENTLOG_{TIMESTAMP}.log
└── Analysis/                      # Analysis scripts and results

Where:
- {PHI_VALUE}: Flux bias values (e.g., 0p300, 0p322, etc.)
- {DA}: DA value (e.g., 16, 18, 20, etc.)
- {SR}: SR value (e.g., 10)
- {FREQ}: Frequency in GHz (e.g., 5p00, 8p00, 11p00, etc.)
- {POWER}: Power in dBm (e.g., -10p0, -7p0, -1p0, 2p0, etc.)
- {DATE}: Date in MMDDYY format
- {RUN_NAME}: Run identifier (e.g., L1A_RUN1)
- {TIMESTAMP}: Timestamp in YYYYMMDD_HHMMSS format
```
