# MR Thermometry

This guide covers MR thermometry methods using the `qmri.thermometry` module.

## Overview

MR thermometry enables non-invasive temperature measurement. The `qmri.thermometry` module provides two methods:

- **Multi-echo dual-resonance**: Absolute temperature from ethylene glycol phantoms
- **PRF shift**: Relative temperature changes for in-vivo thermal therapy monitoring

## Multi-Echo Dual-Resonance Thermometry

The multi-echo method is designed for **ethylene glycol phantom thermometry**, providing absolute temperature measurements from multi-echo magnitude data.

### Overview

Unlike PRF thermometry which measures relative temperature changes from phase, the dual-resonance method:

- Uses **magnitude** data (not phase)
- Provides **absolute** temperature (not relative change)
- Is calibrated for **ethylene glycol** phantoms
- Requires **multiple echoes** to fit the signal model

This method is ideal for phantom calibration and validation studies.

### Physical Principle

Ethylene glycol contains two distinct chemical species (CH₂ and OH groups) that produce a characteristic beat pattern in the multi-echo signal. The frequency of this beat pattern depends on temperature, allowing absolute temperature measurement.

### Basic Usage

```python
import numpy as np
from qmri.thermometry import multiecho

# Example: Fit synthetic data at known temperature
magnetic_field = 3.0  # Tesla
temperature_true = 25.0  # Celsius

# Generate echo times (24 echoes from 1-24 ms)
echo_times = np.linspace(0.001, 0.024, 24)

# Calculate expected frequency difference for this temperature
df = multiecho.calculate_df_from_temperature(temperature_true, magnetic_field)

# Generate synthetic signal
signal = multiecho.thermometry_signal_model(
    t=echo_times,
    amplitude_1=1.0,
    amplitude_2=0.5,
    r2star_1=30.0,
    r2star_2=40.0,
    df=df,
    dphi_deg=0.0,
)

# Fit the model to recover temperature
result = multiecho.fit_multiecho_thermometry(
    signal, echo_times, magnetic_field, method="single"
)

print(f"Fitted temperature: {result.temperature:.1f} °C")
print(f"True temperature: {temperature_true:.1f} °C")
print(f"R²: {result.r_squared:.4f}")
```

### Signal Model Visualisation

The dual-resonance signal shows a characteristic oscillating decay:

```python
import numpy as np
import matplotlib.pyplot as plt
from qmri.thermometry import multiecho

# Parameters
echo_times = np.linspace(0.001, 0.050, 200)
magnetic_field = 3.0

# Generate signals at different temperatures
temperatures = [20, 25, 30, 35]
plt.figure(figsize=(10, 6))

for temp in temperatures:
    df = multiecho.calculate_df_from_temperature(temp, magnetic_field)
    signal = multiecho.thermometry_signal_model(
        echo_times, 1.0, 0.5, 30.0, 40.0, df, 0.0
    )
    plt.plot(echo_times * 1000, signal, label=f"{temp}°C")

plt.xlabel("Echo Time (ms)")
plt.ylabel("Signal")
plt.title("Dual-Resonance Signal at Different Temperatures")
plt.legend()
plt.grid(True)
plt.show()
```

### Temperature-Frequency Conversion

The calibration is specific to ethylene glycol:

```python
from qmri.thermometry import multiecho

# Convert temperature to frequency difference
temperature = 25.0
b0 = 3.0  # Tesla

df = multiecho.calculate_df_from_temperature(temperature, b0)
print(f"At {temperature}°C and {b0}T: Δf = {df:.1f} Hz")

# Convert back
recovered_temp = multiecho.calculate_temperature_from_df(df, b0)
print(f"Recovered temperature: {recovered_temp:.1f}°C")
```

### MultiEchoResult Dataclass

```python
@dataclass(frozen=True)
class MultiEchoResult:
    temperature: float              # Fitted temperature in °C
    temperature_uncertainty: float  # Uncertainty in °C
    df: float                       # Fitted frequency difference in Hz
    r_squared: float                # Coefficient of determination
    fitted_params: NDArray          # All fitted parameters
    n_bootstrap: int | None         # Number of bootstrap samples (if used)
```

### Fit Quality Assessment

Always check the R² value to assess fit quality:

```python
result = multiecho.fit_multiecho_thermometry(signal, echo_times, magnetic_field)

if result.r_squared > 0.95:
    print(f"Good fit: Temperature = {result.temperature:.1f}°C")
elif result.r_squared > 0.9:
    print(f"Acceptable fit: Temperature = {result.temperature:.1f}°C")
else:
    print("Poor fit - check data quality")
```

### Segmentation-Driven Image Fitting

For whole-image phantom data, `fit_multiecho_thermometry_image` fits the
dual-resonance model over a segmented volume. The segmentation defines discrete
regions by integer label (label `0` is background and is ignored), and each
non-zero region is fitted and converted to temperature.

```python
import numpy as np
from qmri.thermometry import multiecho

# 4D multi-echo magnitude volume (nx, ny, nz, n_echoes) and a 3D label map
signal = ...          # shape (nx, ny, nz, n_echoes)
segmentation = ...    # shape (nx, ny, nz), integer labels
echo_times = np.linspace(0.001, 0.024, 24)  # seconds

temperature_map, regions = multiecho.fit_multiecho_thermometry_image(
    signal,
    segmentation,
    echo_times,
    magnetic_field_tesla=3.0,
    method="regionwise",
)

for region in regions:
    print(
        f"region {region.region_id} "
        f"({region.region_size} voxels): "
        f"{region.temperature:.2f} ± {region.temperature_uncertainty:.2f} °C"
    )
```

The `method` argument selects how each region is summarised:

| Method | Behaviour |
|--------|-----------|
| `regionwise` | Fit the mean signal of each region once; uncertainty from the fitted Δf covariance. Fastest. |
| `voxelwise` | Fit every voxel independently; region summary is the inverse-variance weighted mean of voxel temperatures. |
| `regionwise_bootstrap` | Resample region voxels with replacement, fit each sample's mean signal, and summarise with the mean and standard deviation over samples passing the R² threshold. |

Each entry in the returned list is a `RegionThermometryResult` with per-fit
temperatures, uncertainties, R² values and fitted parameters, plus a
`to_dict()` method for JSON serialisation.

!!! note "Robust starting values (`df_init`)"
    The non-linear fit must be seeded with a starting frequency, and the
    dual-resonance magnitude signal has local minima spaced roughly one over the
    echo-train span apart — so a poor seed can converge to the wrong frequency
    alias (e.g. a 10 °C phantom mis-fitted as ~185 °C on aliasing-prone grids).
    The `df_init` argument (on `fit_multiecho_thermometry`,
    `fit_multiecho_thermometry_image` and the pipeline) selects the strategy:

    - `"multistart"` (default): fit from both a fixed default and a data-driven
      Lomb-Scargle estimate of the beat frequency (from the envelope-removed
      signal), keeping the higher-R² result. Most robust.
    - `"fixed"`: a single fit from the fixed default — cheapest, adequate for
      well-conditioned acquisitions but can alias on cold phantoms.
    - `"lombscargle"`: a single fit seeded from the Lomb-Scargle estimate.

    On well-sampled real phantom data all three typically agree to a few tenths
    of a degree; `multistart` is the default as cheap insurance against aliasing
    on other echo-time schemes.

## End-to-End Pipeline

The `qmri-pipelines` package wraps the segmentation-driven fit into a
file-in / file-out workflow. It loads multi-echo NIfTIs, concatenates and sorts
echoes by echo time, detects the field strength from JSON sidecars, fits, and
writes a temperature-map NIfTI plus a JSON report.

```python
from qmri.pipelines.thermometry import run_multiecho_thermometry

temperature_map, report = run_multiecho_thermometry(
    multiecho_files=["echo_block_1.nii.gz", "echo_block_2.nii.gz"],
    segmentation_file="labels.nii.gz",
    echo_times_files=["te_block_1.txt", "te_block_2.txt"],
    method="regionwise",
    output_dir="results/",
)

for region in report.regions:
    print(f"region {region.region_id}: {region.temperature:.2f} °C")
```

The magnetic field strength is read from a JSON sidecar (`ImagingFrequency` or
`MagneticFieldStrength`) next to the first image unless `magnetic_field_tesla`
is supplied explicitly.

The same pipeline is available from the command line — see the
[CLI Reference](../cli.md#qmri-thermometry-multiecho):

```bash
qmri thermometry multiecho echo_block_1.nii.gz echo_block_2.nii.gz \
    -e te_block_1.txt -e te_block_2.txt \
    -s labels.nii.gz --method regionwise -o results/
```

### When to Use Multi-Echo vs PRF

| Use Case | Method |
|----------|--------|
| In-vivo thermal therapy monitoring | PRF |
| Phantom temperature calibration | Multi-echo |
| Relative temperature change | PRF |
| Absolute temperature measurement | Multi-echo |
| Phase-sensitive sequences | PRF |
| Magnitude-only data | Multi-echo |

### Best Practices

1. **Use sufficient echo times**: At least 20+ echoes spanning multiple beat periods
2. **Check R² values**: Values > 0.95 indicate good fits
3. **Verify field strength**: Ensure correct B0 value is used
4. **Temperature range**: The ethylene glycol calibration is valid from ~10°C to ~60°C
5. **SNR requirements**: Higher SNR improves fitting accuracy

### References

1. Sprinkhuizen, S.M., Bakker, C.J.G. and Bartels, L.W. "Absolute MR thermometry using time-domain analysis of multi-gradient-echo magnitude images." MRM 64:239-248, 2010.

2. Raiford, D.S., Fisk, C.L. and Becker, E.D. "Calibration of methanol and ethylene glycol nuclear magnetic resonance thermometers." Anal. Chem. 51(12):2050-2051, 1979.

---

## PRF Thermometry

The PRF shift method is the most widely used technique for MR-guided thermal interventions.

### Physical Principle

The PRF method exploits the temperature dependence of the water proton resonance frequency. As temperature increases:

1. Hydrogen bonds between water molecules break
2. Electron screening of water protons decreases
3. The local magnetic field experienced by protons changes
4. The resonance frequency shifts (decreases with increasing temperature)

This frequency shift manifests as a phase change in gradient echo images.

### Key Equation

The temperature-induced phase shift is:

$$
\Delta\phi = -\gamma \alpha B_0 \Delta T \cdot TE
$$

And the temperature change can be calculated from the phase difference:

$$
\Delta T = \frac{\Delta\phi}{\gamma \alpha B_0 \cdot TE}
$$

where:

- $\Delta\phi$ is the phase difference (radians)
- $\gamma$ is the gyromagnetic ratio (42.58 MHz/T for protons)
- $\alpha$ is the PRF thermal coefficient (~-0.01 ppm/degC)
- $B_0$ is the magnetic field strength (T)
- $TE$ is the echo time (s)
- $\Delta T$ is the temperature change (degC)

### PRF Thermal Coefficient

The PRF thermal coefficient for water is approximately **-0.01 ppm/degC**. The negative sign indicates that:

- Increasing temperature causes a negative phase shift
- The resonance frequency decreases with increasing temperature

```python
from qmri.thermometry.prf import PRF_THERMAL_COEFFICIENT

print(f"PRF coefficient: {PRF_THERMAL_COEFFICIENT * 1e6:.3f} ppm/°C")
# PRF coefficient: -0.010 ppm/°C
```

## Calculating Temperature from Phase

### Basic Usage

```python
import numpy as np
from qmri.thermometry import prf

# Measured phase difference (radians)
# Negative phase = temperature increase
phase_difference = -0.16  # radians

# Calculate temperature change
result = prf.calculate_temperature(
    phase_difference=phase_difference,
    echo_time=0.020,      # 20 ms
    magnetic_field=3.0,   # 3 Tesla
)

print(f"Temperature change: {result.temperature_change:.1f} °C")
```

### With Baseline Temperature

By default, the calculation assumes a baseline temperature of 37degC (body temperature). You can specify a different baseline:

```python
result = prf.calculate_temperature(
    phase_difference=-0.16,
    echo_time=0.020,
    magnetic_field=3.0,
    baseline_temperature=25.0  # Room temperature baseline
)

absolute_temperature = 25.0 + result.temperature_change
print(f"Absolute temperature: {absolute_temperature:.1f} °C")
```

### Volume Processing

The functions handle multi-dimensional data:

```python
import numpy as np
from qmri.thermometry import prf

# Phase maps from baseline and heated states
phase_baseline = np.random.rand(128, 128) * 2 * np.pi - np.pi
phase_heated = phase_baseline - 0.1  # Simulated heating

# Calculate phase difference
phase_diff = phase_heated - phase_baseline

# Calculate temperature map
result = prf.calculate_temperature(
    phase_difference=phase_diff,
    echo_time=0.015,
    magnetic_field=1.5
)

print(f"Temperature map shape: {result.temperature_change.shape}")
print(f"Mean temperature change: {np.nanmean(result.temperature_change):.1f} °C")
```

## Simulating Phase Shifts

Use `signal_phase_shift()` to calculate the expected phase shift for a given temperature change:

```python
import numpy as np
from qmri.thermometry import prf

# Calculate phase shift for various temperature changes
temperatures = np.array([1, 5, 10, 20, 30])  # °C

phase_shifts = prf.signal_phase_shift(
    temperature_change=temperatures,
    echo_time=0.020,      # 20 ms
    magnetic_field=3.0    # 3T
)

for t, phi in zip(temperatures, phase_shifts):
    print(f"ΔT = {t:2d} °C → Δφ = {np.degrees(phi):6.2f}° ({phi:.3f} rad)")
```

Expected output at 3T with TE=20ms:

| Temperature Change | Phase Shift |
|-------------------|-------------|
| 1 degC | ~0.8 deg |
| 5 degC | ~4 deg |
| 10 degC | ~8 deg |
| 20 degC | ~16 deg |

## PRFResult Dataclass

```python
@dataclass(frozen=True)
class PRFResult:
    temperature_change: NDArray[np.floating]   # Temperature change in °C
    phase_difference: NDArray[np.floating]     # Phase difference in radians
```

## Considerations for Accuracy

### 1. Phase Wrapping

Phase values are typically wrapped to [-pi, pi]. For large temperature changes (>20-30degC at typical parameters), phase unwrapping may be required:

```python
import numpy as np

def unwrap_phase_2d(wrapped_phase):
    """Simple 2D phase unwrapping."""
    return np.unwrap(np.unwrap(wrapped_phase, axis=0), axis=1)

# Apply unwrapping before temperature calculation
unwrapped_diff = unwrap_phase_2d(phase_heated - phase_baseline)
result = prf.calculate_temperature(unwrapped_diff, echo_time=0.02, magnetic_field=3.0)
```

### 2. Fat Tissue

**Important**: Fat does not exhibit PRF temperature sensitivity because:

- Fat protons have different chemical environment
- Hydrogen bonding structure differs from water

This means:

- Fat regions show no temperature change (useful as reference)
- Fat suppression or water-fat separation is often used
- Fat can be used for drift correction

### 3. Field Drift

B0 field drift over time causes apparent temperature changes:

```python
# Example: Using a reference region for drift correction
def correct_drift(temp_map, reference_mask):
    """Correct field drift using a reference region."""
    drift = np.nanmean(temp_map[reference_mask])
    return temp_map - drift

# Use subcutaneous fat or an unheated region as reference
corrected_temp = correct_drift(result.temperature_change, fat_mask)
```

### 4. Motion

Motion between baseline and heated acquisitions causes artefacts. Strategies include:

- **Registration**: Align images before subtraction
- **Multi-baseline**: Acquire multiple baselines and select best match
- **Referenceless methods**: Estimate background phase from heated image

### 5. Susceptibility Effects

Magnetic susceptibility changes with temperature, causing additional phase shifts. This is typically a small effect but may be relevant for:

- High-precision measurements
- Interfaces between tissues
- Regions near air-tissue boundaries

### 6. Echo Time Selection

The choice of TE involves a trade-off:

| Longer TE | Shorter TE |
|-----------|------------|
| Larger phase shift | Smaller phase shift |
| Lower SNR | Higher SNR |
| More T2* decay | Less T2* decay |
| Better temperature sensitivity | More robust signal |

**Recommendation**: TE ~ 10-20 ms at 1.5T-3T provides a good balance.

## Complete Example: Thermal Ablation Monitoring

```python
import numpy as np
from qmri.thermometry import prf

# Acquisition parameters
te = 0.012  # 12 ms
b0 = 3.0    # 3 Tesla

# Simulate a focal heating pattern
x = np.linspace(-32, 32, 128)
y = np.linspace(-32, 32, 128)
X, Y = np.meshgrid(x, y)

# Gaussian heating pattern (focal ablation)
sigma = 5  # mm
peak_temp_rise = 30  # °C
temp_true = peak_temp_rise * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Calculate expected phase shift
phase_shift_true = prf.signal_phase_shift(
    temperature_change=temp_true,
    echo_time=te,
    magnetic_field=b0
)

# Add noise (typical phase noise ~0.05 rad)
noise_std = 0.05
phase_shift_noisy = phase_shift_true + np.random.normal(0, noise_std, phase_shift_true.shape)

# Recover temperature
result = prf.calculate_temperature(
    phase_difference=phase_shift_noisy,
    echo_time=te,
    magnetic_field=b0,
    baseline_temperature=37.0
)

# Calculate statistics
temp_recovered = result.temperature_change
max_temp = np.max(temp_recovered)
max_true = np.max(temp_true)

print(f"Peak temperature rise:")
print(f"  True: {max_true:.1f} °C")
print(f"  Measured: {max_temp:.1f} °C")
print(f"  Error: {max_temp - max_true:+.1f} °C")

# Calculate temperature uncertainty from phase noise
# dT/dphi = 1 / (gamma * alpha * B0 * TE)
temp_uncertainty = noise_std / (2 * np.pi * 42.58e6 * (-0.01e-6) * b0 * te)
print(f"Temperature uncertainty (1 SD): ~{abs(temp_uncertainty):.1f} °C")
```

## Temperature Sensitivity Analysis

```python
import numpy as np
from qmri.thermometry import prf

# Compare sensitivity at different field strengths and echo times
field_strengths = [1.5, 3.0, 7.0]  # Tesla
echo_times = [0.010, 0.015, 0.020]  # seconds

print("Phase shift (degrees) for 10°C temperature change:")
print("-" * 50)
print(f"{'TE (ms)':<10}", end="")
for b0 in field_strengths:
    print(f"{b0:.1f}T{'':<8}", end="")
print()

for te in echo_times:
    print(f"{te*1000:<10.0f}", end="")
    for b0 in field_strengths:
        phi = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=te,
            magnetic_field=b0
        )
        print(f"{np.degrees(phi):<12.1f}", end="")
    print()
```

## Best Practices

### Acquisition

1. **Use gradient echo sequences** (spoiled GRE recommended).

2. **Select appropriate TE**: 10-20 ms provides good sensitivity.

3. **Acquire fat-suppressed or water-selective images** if possible.

4. **Include reference regions** (unheated tissue or fat) for drift correction.

5. **Use fast imaging** for real-time monitoring (EPI, spiral).

### Processing

1. **Always correct for field drift** using reference regions.

2. **Apply spatial filtering** to reduce noise (but preserves edges).

3. **Use phase unwrapping** for large temperature changes.

4. **Mask regions with low SNR** (typically where magnitude < threshold).

5. **Monitor for motion** and re-register if necessary.

### Quality Control

1. **Verify baseline stability** before heating.

2. **Check reference region temperatures** remain constant.

3. **Monitor for phase wrapping** in regions of rapid temperature change.

4. **Validate with fibre optic thermometry** when possible.

## Typical Temperature Sensitivity

At 3T with TE = 15 ms:

- Phase shift: ~1.2 deg/degC
- With phase noise of 5 deg: Temperature uncertainty ~4degC
- With phase noise of 1 deg: Temperature uncertainty ~0.8degC

## References

1. Ishihara, Y., et al. "A precise and fast temperature mapping using water proton chemical shift." MRM 34(6):814-823, 1995.

2. Rieke, V. and Butts Pauly, K. "MR thermometry." J Magn Reson Imaging 27(2):376-390, 2008.

3. De Poorter, J., et al. "Noninvasive MRI thermometry with the proton resonance frequency method: study of susceptibility effects." MRM 34(3):359-367, 1995.

4. Quesson, B., et al. "Magnetic resonance temperature imaging for guidance of thermotherapy." J Magn Reson Imaging 12(4):525-533, 2000.
