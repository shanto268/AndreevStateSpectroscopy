# Andreev State Spectroscopy

This repository contains the code for the Andreev State Spectroscopy project.

## Analysis Notebooks:

### HMMAnalysis.ipynb

**Expected data structure:**

```bash
Quasiparticles/
├── Andreev_Spectroscopy/
│   └── {DATE}/                    # Format: MMDDYY
│       └── {RUN_NAME}/            # e.g., L1A_RUN1
│           ├── phi_{PHI_VALUE}/   # Format: phi_0p300, phi_0p322, etc.
│           │   └── DA{DA}_SR{SR}/ # Format: DA16_SR10, DA18_SR10, etc.
│           │       └── {RUN_NAME}_{TIMESTAMP}.bin  # Binary data file
│           │       └── {RUN_NAME}_{TIMESTAMP}.txt  # Metadata file
│           ├── Figures/           # Generated figures
│           ├── vna_fit_{PHI}.pkl  # VNA fit data for each phi value
│           └── MEASUREMENTLOG_{TIMESTAMP}.log  # DAQ metadata file
└── Analysis/                      # Analysis scripts and results

Where:
- {PHI_VALUE}: Flux bias values (e.g., 0p300, 0p322, etc.)
- {DA}: DA value (e.g., 16, 18, 20, etc.)
- {SR}: SR value (e.g., 10)
- {DATE}: Date in MMDDYY format
- {RUN_NAME}: Run identifier (e.g., L1A_RUN1)
- {TIMESTAMP}: Timestamp in YYYYMMDD_HHMMSS format
```

## HPC Workflow:

### Data Structure (RUN01)

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

### Work to do:

- [ ] Store power to the device in the results file
- [ ] Append the DAQ metadata to the results file
- [ ] Get the list of non-linear-attenuations for our test dataset
- [ ] Create the means getter flow for no clearing tone case
- [ ] Create and run the means getter workflow for the clearing tone case
- [ ] Add the saving of clearing tone parameters to the results file
- [ ] Run the HMM workflow in a parallelized way
  - 1 node per $(\phi, DA, f_c, P_c)$
- [ ] Figures of merit:
  - Rates, mean occupation, probabilities against as a function of $\phi$
