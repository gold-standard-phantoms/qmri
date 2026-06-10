# Diffusion Imaging

This guide covers Apparent Diffusion Coefficient (ADC) fitting from diffusion-weighted MRI data using the `qmri.diffusion` module.

## Overview

Diffusion-weighted imaging (DWI) measures the random motion of water molecules in tissue. The signal attenuation follows the Stejskal-Tanner equation:

$$
S(b) = S_0 \exp(-b \cdot \text{ADC})
$$

where:

- $S(b)$ is the signal at diffusion weighting $b$
- $S_0$ is the baseline signal (at $b=0$)
- $b$ is the diffusion weighting factor (s/mm²)
- ADC is the Apparent Diffusion Coefficient (mm²/s)

## ADC Fitting Methods

The `qmri.diffusion.adc` module provides three fitting methods, each with different trade-offs between speed and accuracy.

### Linear Least Squares (LLS)

The simplest and fastest method. It linearises the problem by taking the logarithm:

$$
\ln(S) = \ln(S_0) - b \cdot \text{ADC}
$$

```python
import numpy as np
from qmri.diffusion import adc

b_values = np.array([0, 500, 1000, 2000])
signal = np.array([1000, 606, 368, 135])

result = adc.fit(signal, b_values, method="lls")
print(f"ADC: {result.adc:.2e} mm²/s")
```

**When to use**: Quick initial estimates, high SNR data, or when processing speed is critical.

**Limitations**: The log transform amplifies noise at low signal values (high b-values), leading to biased estimates in noisy data.

### Weighted Linear Least Squares (WLLS)

Improves upon LLS by applying weights based on the expected signal intensity:

$$
\hat{\boldsymbol{\beta}}_{WLLS} = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{y}
$$

where $\mathbf{W} = \text{diag}(\exp(2\mathbf{X}\hat{\beta}_{LLS}))$.

```python
result = adc.fit(signal, b_values, method="wlls")
```

**When to use**: Moderate SNR data where you need better accuracy than LLS without the computational cost of IWLLS.

### Iterative Weighted Linear Least Squares (IWLLS)

The recommended method. It iteratively refines the weights until convergence:

```python
result = adc.fit(
    signal,
    b_values,
    method="iwlls",
    max_iterations=10,
    tolerance=1e-6
)
print(f"ADC: {result.adc:.2e} mm²/s")
print(f"Iterations: {result.iterations}")
```

**When to use**: Default choice for most applications. Provides the most accurate estimates, particularly for noisy data.

**Parameters**:

- `max_iterations`: Maximum number of iterations (default: 10)
- `tolerance`: Convergence threshold for ADC change (default: 1e-6)

Typically converges in 2-3 iterations.

## Choosing a Method

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| LLS | Fast | Moderate | High SNR, quick estimates |
| WLLS | Medium | Good | Moderate SNR |
| IWLLS | Slower | Best | General use, low SNR |

For clinical data, **IWLLS is recommended** as it provides robust estimates across varying SNR levels.

## Working with Multi-Voxel Data

The `fit()` function automatically handles both single-voxel and volumetric data.

### Single Voxel

```python
import numpy as np
from qmri.diffusion import adc

b_values = np.array([0, 500, 1000, 2000])
signal = np.array([1000, 606, 368, 135])

result = adc.fit(signal, b_values, method="iwlls")
# Returns ADCResult with scalar values
print(f"ADC: {result.adc:.2e} mm²/s")
print(f"S0: {result.s0:.0f}")
print(f"R²: {result.r_squared:.4f}")
```

### Volume Processing

For 4D DWI data where the last dimension contains b-values:

```python
import numpy as np
from qmri.diffusion import adc

# Simulated 4D DWI data: (x, y, z, b-values)
dwi_4d = np.random.rand(64, 64, 30, 4) * 1000
b_values = np.array([0, 500, 1000, 2000])

result = adc.fit(dwi_4d, b_values, method="iwlls")
# Returns ADCMapResult with 3D arrays
print(f"ADC map shape: {result.adc.shape}")  # (64, 64, 30)
print(f"S0 map shape: {result.s0.shape}")    # (64, 64, 30)
print(f"R² map shape: {result.r_squared.shape}")  # (64, 64, 30)
```

### Using a Mask

To process only specific voxels (e.g., within a brain mask):

```python
# Create or load a binary mask
mask = np.zeros((64, 64, 30), dtype=bool)
mask[10:54, 10:54, 5:25] = True  # Process only central region

result = adc.fit(dwi_4d, b_values, method="iwlls", mask=mask)
# Voxels outside the mask will have zero values
```

## Understanding the ADCResult Dataclass

### Single Voxel Result

For 1D signal input, `fit()` returns an `ADCResult`:

```python
@dataclass(frozen=True)
class ADCResult:
    adc: float           # ADC in mm²/s
    s0: float            # Baseline signal intensity
    r_squared: float     # Coefficient of determination (0-1)
    iterations: int | None  # Number of iterations (IWLLS only)
```

**Interpreting the fields**:

- `adc`: The fitted Apparent Diffusion Coefficient. Typical values for brain tissue are 0.7-1.0 × 10⁻³ mm²/s.
- `s0`: The extrapolated signal at b=0, representing proton density weighted contrast.
- `r_squared`: Fit quality metric. Values close to 1 indicate a good fit. Values below 0.9 may indicate poor data quality or non-mono-exponential decay.
- `iterations`: Only populated for IWLLS; shows convergence behaviour.

### Volume Result

For multi-dimensional input, `fit()` returns an `ADCMapResult`:

```python
@dataclass(frozen=True)
class ADCMapResult:
    adc: NDArray[np.floating]        # ADC map in mm²/s
    s0: NDArray[np.floating]         # Baseline signal map
    r_squared: NDArray[np.floating]  # R² quality map
```

Each array has the same spatial dimensions as the input (excluding the b-value dimension).

## Generating Synthetic DWI Signal

Use `signal_model()` to generate synthetic DWI data for testing or simulation:

```python
from qmri.diffusion import adc
import numpy as np

b_values = np.array([0, 200, 500, 800, 1000, 1500, 2000])
signal = adc.signal_model(s0=1000, adc=1e-3, b_values=b_values)

print(signal.round(0))
# [1000.  819.  607.  449.  368.  223.  135.]
```

This is useful for:

- Validating fitting algorithms
- Creating phantoms for testing pipelines
- Simulating the effect of different acquisition parameters

## Complete Example

Here is a complete workflow demonstrating ADC fitting with quality assessment:

```python
import numpy as np
from qmri.diffusion import adc

# Acquisition parameters
b_values = np.array([0, 200, 500, 800, 1000, 1500, 2000])

# Generate synthetic data with known ADC
true_adc = 1.0e-3  # mm²/s (typical for grey matter)
true_s0 = 1000
clean_signal = adc.signal_model(s0=true_s0, adc=true_adc, b_values=b_values)

# Add Rician noise (more realistic for magnitude MRI)
noise_level = 30
real_noise = np.random.normal(0, noise_level, clean_signal.shape)
imag_noise = np.random.normal(0, noise_level, clean_signal.shape)
noisy_signal = np.sqrt((clean_signal + real_noise)**2 + imag_noise**2)

# Fit using different methods
methods = ["lls", "wlls", "iwlls"]
for method in methods:
    result = adc.fit(noisy_signal, b_values, method=method)
    error = 100 * (result.adc - true_adc) / true_adc
    print(f"{method.upper():6s}: ADC = {result.adc:.4e} mm²/s "
          f"(error: {error:+.1f}%), R² = {result.r_squared:.4f}")
```

## Best Practices

1. **Use IWLLS** as your default method unless speed is critical.

2. **Include a b=0 image** for reliable S₀ estimation.

3. **Use multiple b-values** (at least 4) for robust fitting.

4. **Check R² values** to identify voxels with poor fit quality.

5. **Apply a mask** to exclude background and non-tissue voxels.

6. **Consider the b-value range**:
   - Low b-values (0-500 s/mm²): Sensitive to perfusion contamination
   - High b-values (>1500 s/mm²): Lower SNR, but better ADC specificity

## References

1. Veraart, J., et al. (2013). "Weighted linear least squares estimation of diffusion MRI parameters: Strengths, limitations, and pitfalls." NeuroImage 81:335-346.

2. Basser, P.J., Mattiello, J., Le Bihan, D. (1994). "Estimation of the effective self-diffusion tensor from the NMR spin echo." J Magn Reson B 103(3):247-254.
