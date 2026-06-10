# Tutorials

Interactive Jupyter notebooks demonstrating qmri workflows.

## Try Online

Launch the tutorials directly in your browser with Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gold-standard-phantoms/qmri/main?labpath=examples%2Fjupyter)

## Available Notebooks

### 1. ADC Fitting Workflow

**File:** `01_adc_fitting_workflow.ipynb`

Learn how to:

- Generate synthetic DWI data with known ADC
- Fit ADC using different methods
- Visualise fitting results and error maps
- Understand clinical ADC values

### 2. T1 Mapping with Synthetic Data

**File:** `02_t1_mapping_synthetic.ipynb`

Learn how to:

- Generate inversion recovery (IR) data
- Compare general vs classical IR models
- Use variable TR (VTR) method
- Create multi-voxel T1 maps

### 3. ASL Perfusion Quantification

**File:** `03_asl_perfusion_quantification.ipynb`

Learn how to:

- Generate pCASL control and label images
- Understand the General Kinetic Model
- Explore transit time effects
- Visualise perfusion maps

### 4. Method Comparison Benchmark

**File:** `04_method_comparison_benchmark.ipynb`

Learn how to:

- Compare LLS, WLLS, and IWLLS fitting
- Run Monte Carlo simulations
- Analyse bias and variance trade-offs
- Benchmark different b-value protocols

### 5. Noise Sensitivity Analysis

**File:** `05_noise_sensitivity_analysis.ipynb`

Learn how to:

- Compare Gaussian vs Rician noise
- Determine SNR requirements
- Optimise acquisition design
- Quantify parameter uncertainty

## Running Locally

Clone the repository and run the notebooks locally:

```bash
# Clone repository
git clone https://github.com/gold-standard-phantoms/qmri.git
cd qmri

# Install dependencies
uv sync

# Start JupyterLab
uv run jupyter lab examples/jupyter/
```

## Prerequisites

The tutorials assume familiarity with:

- Python and NumPy basics
- MRI physics fundamentals
- Jupyter notebook interface

No prior qmri experience is required.
