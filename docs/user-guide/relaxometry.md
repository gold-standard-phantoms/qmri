# Relaxometry

This guide covers T1 and T2 mapping using the `qmri.relaxometry` module.

## Overview

Relaxometry measures the characteristic relaxation times of tissue:

- **T1 (spin-lattice relaxation)**: Time constant for longitudinal magnetisation recovery
- **T2 (spin-spin relaxation)**: Time constant for transverse magnetisation decay

These parameters are fundamental tissue properties that provide contrast in MRI and can be quantified for diagnostic purposes.

## T1 Mapping

The `qmri.relaxometry.t1` module provides two methods for T1 mapping:

- **Inversion Recovery (IR)**: The gold standard, using inversion pulses
- **Variable TR (VTR)**: Faster alternative using saturation recovery

### Inversion Recovery Methods

#### Signal Models

**General IR Model** (includes TR recovery):

$$
S = S_0 \left(1 - 2\alpha \exp\left(-\frac{TI}{T_1}\right) + \exp\left(-\frac{TR}{T_1}\right)\right)
$$

**Classical IR Model** (assumes TR >> T1):

$$
S = S_0 \left(1 - 2\alpha \exp\left(-\frac{TI}{T_1}\right)\right)
$$

where:

- $S_0$ is the equilibrium signal amplitude
- $TI$ is the inversion time (s)
- $TR$ is the repetition time (s)
- $\alpha$ is the inversion efficiency (ideally 1.0)
- $T_1$ is the longitudinal relaxation time (s)

#### Basic T1 IR Fitting

```python
import numpy as np
from qmri.relaxometry import t1

# Inversion times
ti = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])

# Generate synthetic IR signal
signal = t1.signal_ir(
    s0=1000,
    t1=1.2,  # seconds
    inversion_times=ti,
    repetition_times=5.0,
    inversion_efficiency=0.95
)

# Fit T1 using the general model
result = t1.fit_ir(signal, ti, repetition_times=5.0, model="general")
print(f"T1 = {result.t1:.3f} s")
print(f"S0 = {result.s0:.0f}")
print(f"Inversion efficiency = {result.inversion_efficiency:.2f}")
```

#### Choosing Between General and Classical Models

**General model** (`model="general"`):

- Use when TR is comparable to T1 (TR < 5*T1)
- More accurate but has an additional parameter
- Recommended for most applications

**Classical model** (`model="classical"`):

- Use when TR >> T1 (typically TR > 5*T1)
- Simpler, faster fitting
- Suitable for long TR acquisitions

```python
# Classical model (assumes TR >> T1)
result = t1.fit_ir(signal, ti, repetition_times=10.0, model="classical")
```

### Variable TR (VTR) Method

The VTR method uses saturation recovery with varying repetition times:

$$
S = M \left(1 - \exp\left(-\frac{TR}{T_1}\right)\right)
$$

where $M$ is the equilibrium magnetisation.

```python
import numpy as np
from qmri.relaxometry import t1

# Repetition times
tr = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# Generate synthetic VTR signal
signal = t1.signal_vtr(m=1000, t1=1.2, repetition_times=tr)

# Fit T1
result = t1.fit_vtr(signal, tr)
print(f"T1 = {result.t1:.3f} s")
print(f"M = {result.m:.0f}")
```

**When to use VTR**:

- Faster acquisition than IR
- Suitable when inversion pulses are problematic
- Lower accuracy than IR for short T1 values

### Convenience Function

The `fit()` function provides a unified interface:

```python
from qmri.relaxometry import t1

# IR fitting
result = t1.fit(signal, ti, method="ir", repetition_times=5.0)

# Classical IR
result = t1.fit(signal, ti, method="ir_classical", repetition_times=5.0)

# VTR fitting
result = t1.fit(signal, tr, method="vtr")
```

### T1 Result Classes

**T1IRResult** (Inversion Recovery):

```python
@dataclass(frozen=True)
class T1IRResult:
    t1: NDArray[np.floating]                  # T1 in seconds
    s0: NDArray[np.floating]                  # Equilibrium signal
    inversion_efficiency: NDArray[np.floating]  # Inversion efficiency
```

**T1VTRResult** (Variable TR):

```python
@dataclass(frozen=True)
class T1VTRResult:
    t1: NDArray[np.floating]  # T1 in seconds
    m: NDArray[np.floating]   # Equilibrium magnetisation
```

## T2 Mapping

The `qmri.relaxometry.t2` module provides T2 fitting from multi-echo spin echo (MESE) data.

### Signal Model

**Full model** (with offset):

$$
S(TE) = A \exp\left(-\frac{TE}{T_2}\right) + C
$$

**Reduced model** (no offset):

$$
S(TE) = A \exp\left(-\frac{TE}{T_2}\right)
$$

where:

- $A$ is the signal amplitude
- $TE$ is the echo time (s)
- $T_2$ is the transverse relaxation time (s)
- $C$ is an offset term (noise floor, Rician bias)

### Basic T2 Fitting

```python
import numpy as np
from qmri.relaxometry import t2

# Echo times
te = np.array([0.010, 0.020, 0.040, 0.060, 0.080, 0.100, 0.120])

# Generate synthetic T2 decay
signal = t2.signal_decay(amplitude=1000, t2=0.080, echo_times=te)

# Fit T2 with offset
result = t2.fit(signal, te, model="full")
print(f"T2 = {result.t2 * 1000:.1f} ms")
print(f"Amplitude = {result.amplitude:.0f}")
print(f"Offset = {result.offset:.1f}")
```

### Choosing Between Full and Reduced Models

**Full model** (`model="full"`):

- Includes offset term to account for noise floor
- More accurate for low SNR data
- Recommended for magnitude images

**Reduced model** (`model="reduced"`):

- Simpler two-parameter fit
- Suitable for high SNR or phase-corrected data
- Faster computation

### Skipping Early Echoes

The first echo(es) often exhibit signal contamination from stimulated echoes and imperfect refocusing. Skipping these improves T2 accuracy:

```python
# Skip the first echo (default behaviour)
result = t2.fit(signal, te, skip_echoes=1)

# Skip the first two echoes
result = t2.fit(signal, te, skip_echoes=2)

# Use all echoes (not recommended)
result = t2.fit(signal, te, skip_echoes=0)
```

### T2 Result Class

```python
@dataclass(frozen=True)
class T2Result:
    t2: NDArray[np.floating]        # T2 in seconds
    amplitude: NDArray[np.floating]  # Signal amplitude
    offset: NDArray[np.floating] | None  # Offset (full model only)
```

## Volume Processing

Both T1 and T2 fitting functions handle multi-dimensional data automatically.

### T1 Volume Example

```python
import numpy as np
from qmri.relaxometry import t1

# 4D IR data: (x, y, z, inversion_times)
ir_4d = np.random.rand(64, 64, 30, 8) * 1000
ti = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])

# Optional mask
mask = np.ones((64, 64, 30), dtype=bool)
mask[:5, :, :] = False  # Exclude edge slices

# Fit T1 map
result = t1.fit_ir(ir_4d, ti, repetition_times=5.0, mask=mask)
print(f"T1 map shape: {result.t1.shape}")  # (64, 64, 30)
```

### T2 Volume Example

```python
import numpy as np
from qmri.relaxometry import t2

# 4D MESE data: (x, y, z, echo_times)
mese_4d = np.random.rand(64, 64, 30, 10) * 1000
te = np.linspace(0.01, 0.10, 10)

# Fit T2 map
result = t2.fit(mese_4d, te, model="full", skip_echoes=1)
print(f"T2 map shape: {result.t2.shape}")  # (64, 64, 30)
```

## Complete Example: T1 and T2 Mapping

```python
import numpy as np
from qmri.relaxometry import t1, t2

# Tissue parameters (white matter at 3T)
true_t1 = 0.85  # seconds
true_t2 = 0.070  # seconds

# T1 mapping with IR
ti = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
ir_signal = t1.signal_ir(
    s0=1000,
    t1=true_t1,
    inversion_times=ti,
    repetition_times=5.0
)
# Add noise
ir_signal += np.random.normal(0, 20, ir_signal.shape)

t1_result = t1.fit_ir(ir_signal, ti, repetition_times=5.0)
print(f"T1: {t1_result.t1 * 1000:.0f} ms (true: {true_t1 * 1000:.0f} ms)")

# T2 mapping with MESE
te = np.array([0.010, 0.020, 0.030, 0.040, 0.060, 0.080, 0.100])
t2_signal = t2.signal_decay(amplitude=1000, t2=true_t2, echo_times=te)
# Add noise
t2_signal += np.random.normal(0, 20, t2_signal.shape)

t2_result = t2.fit(t2_signal, te, model="full", skip_echoes=1)
print(f"T2: {t2_result.t2 * 1000:.0f} ms (true: {true_t2 * 1000:.0f} ms)")
```

## Best Practices

### T1 Mapping

1. **Choose appropriate TI values**: Sample the recovery curve well, including values near the null point (TI ~ 0.69 * T1).

2. **Use at least 6-8 inversion times** for robust fitting.

3. **Set max_t1 appropriately**: Prevents unrealistic values from noise.
   ```python
   result = t1.fit_ir(signal, ti, repetition_times=5.0, max_t1=5.0)
   ```

4. **Consider polarity restoration**: The fitting automatically handles magnitude IR data by testing different polarity restoration points.

### T2 Mapping

1. **Skip the first echo**: Reduces errors from stimulated echoes.

2. **Use multiple echo times**: At least 5-7 for reliable fitting.

3. **Sample the decay curve appropriately**: Include echo times spanning the expected T2 range.

4. **Use the full model** for magnitude images to account for noise floor.

5. **Set max_t2 appropriately**:
   ```python
   result = t2.fit(signal, te, max_t2=1.0)  # Cap at 1 second
   ```

### Quality Control

1. **Inspect fitted parameter maps** for unrealistic values.

2. **Check boundary voxels** where partial volume effects are common.

3. **Use masks** to exclude background and non-tissue regions.

## Typical Relaxation Times

| Tissue | T1 at 1.5T (ms) | T1 at 3T (ms) | T2 (ms) |
|--------|-----------------|---------------|---------|
| Grey matter | 900-1100 | 1200-1400 | 80-100 |
| White matter | 600-800 | 800-900 | 60-80 |
| CSF | 3000-4000 | 4000-5000 | 1500-2000 |
| Muscle | 800-1000 | 1200-1400 | 30-50 |
| Fat | 200-300 | 300-400 | 50-80 |

## References

1. Barral, J.K., et al. "A robust methodology for in vivo T1 mapping." MRM 64(4):1057-67, 2010.

2. Milford, D., et al. "Mono-Exponential Fitting in T2-Relaxometry: Relevance of Offset and First Echo." PLOS ONE 10(12), 2015.

3. Deoni, S.C., et al. "Determination of optimal angles for variable nutation proton magnetic spin-lattice, T1, and spin-spin, T2, relaxation times measurement." MRM 51(1):194-9, 2004.
