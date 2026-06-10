# Perfusion Imaging

This guide covers Arterial Spin Labelling (ASL) quantification using the `qmri.perfusion` module.

## Overview

Arterial Spin Labelling (ASL) is a non-invasive MRI technique for measuring cerebral blood flow (CBF). It uses magnetically labelled arterial blood water as an endogenous tracer.

The `qmri.perfusion` module provides:

- **ASL quantification** (`qmri.perfusion.asl`): White Paper consensus equations
- **General Kinetic Model** (`qmri.perfusion.gkm`): Full GKM signal generation

## ASL Basics

ASL works by:

1. **Labelling**: Inverting the magnetisation of inflowing arterial blood
2. **Waiting**: Allowing labelled blood to flow into the imaging region
3. **Imaging**: Acquiring "label" and "control" images
4. **Subtraction**: Computing the perfusion-weighted difference signal

The difference signal ($\Delta M = \text{control} - \text{label}$) is proportional to CBF.

## ASL Labelling Schemes

### Pseudo-Continuous ASL (pCASL)

The recommended clinical standard. Uses a train of RF pulses to continuously label blood.

**Key parameters**:

- **Label duration** ($\tau$): Typically 1.5-2.0 s
- **Post-label delay** (PLD): Typically 1.5-2.0 s
- **Labelling efficiency** ($\alpha$): Typically 0.85

### Pulsed ASL (PASL)

Uses a single inversion pulse to create a labelled bolus.

**Key parameters**:

- **Bolus duration** ($TI_1$): Typically 0.7-1.0 s
- **Inversion time** ($TI$): Typically 1.5-2.0 s
- **Labelling efficiency** ($\alpha$): Typically 0.98

## ASL Quantification

### pCASL/CASL Quantification

The White Paper consensus equation for pCASL:

$$
f = \frac{6000 \cdot \lambda \cdot \Delta M \cdot e^{PLD/T_{1,b}}}{2 \cdot \alpha \cdot T_{1,b} \cdot M_0 \cdot (1 - e^{-\tau/T_{1,b}})}
$$

where $f$ is perfusion in ml/100g/min.

```python
import numpy as np
from qmri.perfusion import asl

# Image data
control = np.array([1000.0, 1050.0, 980.0])
label = np.array([950.0, 1000.0, 935.0])
m0 = np.array([2000.0, 2100.0, 1950.0])

# Quantify perfusion
result = asl.quantify_pcasl(
    control=control,
    label=label,
    m0=m0,
    label_duration=1.8,      # seconds
    post_label_delay=1.8,    # seconds
    label_efficiency=0.85,   # typical for pCASL
    t1_blood=1.65,          # seconds at 3T
    partition_coefficient=0.9  # ml/g
)

print(f"CBF: {result.perfusion} ml/100g/min")
```

### PASL Quantification

The White Paper equation for PASL:

$$
f = \frac{6000 \cdot \lambda \cdot \Delta M \cdot e^{TI/T_{1,b}}}{2 \cdot \alpha \cdot TI_1 \cdot M_0}
$$

```python
from qmri.perfusion import asl

result = asl.quantify_pasl(
    control=control,
    label=label,
    m0=m0,
    bolus_duration=0.7,      # TI1 in seconds
    inversion_time=1.8,      # TI in seconds
    label_efficiency=0.98,   # typical for PASL
    t1_blood=1.65,
    partition_coefficient=0.9
)

print(f"CBF: {result.perfusion} ml/100g/min")
```

## General Kinetic Model

The `qmri.perfusion.gkm` module implements the Buxton General Kinetic Model for simulating ASL signals.

### GKM Theory

The GKM calculates the magnetisation difference as:

$$
\Delta M(t) = 2 M_{0,b} f \lbrace c(t) \ast [r(t) \cdot m(t)] \rbrace
$$

where:

- $c(t)$ = delivery function (arterial input)
- $r(t) = e^{-ft/\lambda}$ = residue function
- $m(t) = e^{-t/T_1}$ = magnetisation relaxation

The model accounts for:

- Arterial transit time (ATT)
- Label duration
- T1 relaxation of blood and tissue
- Perfusion rate

### Using the Full GKM

```python
import numpy as np
from qmri.perfusion import gkm

result = gkm.signal_gkm(
    perfusion_rate=60.0,        # ml/100g/min
    transit_time=1.0,           # arterial transit time (s)
    m0_tissue=1000.0,
    label_duration=1.8,         # seconds
    signal_time=3.6,            # PLD + tau
    label_efficiency=0.85,
    partition_coefficient=0.9,
    t1_blood=1.65,
    t1_tissue=1.3,
    label_type="pcasl"          # or "pasl"
)

print(f"Delta M: {result.delta_m}")
```

### Using the Simplified GKM

The simplified version assumes the bolus has fully arrived:

```python
from qmri.perfusion import gkm

result = gkm.signal_gkm_simplified(
    perfusion_rate=60.0,
    transit_time=1.0,
    m0_tissue=1000.0,
    label_duration=1.8,
    signal_time=3.6,
    label_efficiency=0.85,
    partition_coefficient=0.9,
    t1_blood=1.65,
    label_type="pcasl"
)
```

### GKM Arrival States

The GKM models three temporal phases:

1. **Not arrived** ($t \leq \delta$): No labelled blood has reached tissue
2. **Arriving** ($\delta < t < \delta + \tau$): Bolus is arriving
3. **Arrived** ($t \geq \delta + \tau$): Entire bolus has arrived

where $\delta$ is the arterial transit time and $\tau$ is the label duration.

## Parameter Choices and Their Effects

### Label Duration (τ)

| Value | Effect |
|-------|--------|
| Short (1.0 s) | Lower signal, less T1 decay |
| Long (2.0 s) | Higher signal, more T1 decay, longer scan |

**Recommendation**: 1.8 s is a good compromise for pCASL.

### Post-Label Delay (PLD)

| Value | Effect |
|-------|--------|
| Short (1.0 s) | Higher signal but ATT sensitivity |
| Long (2.5 s) | Lower signal but robust to ATT variation |

**Recommendations**:

- Young healthy adults: 1.8 s
- Elderly or vascular disease: 2.0-2.5 s
- Children: 1.5 s

### T1 of Blood (T₁,ᵦ)

Field strength dependent:

| Field | T1 blood (s) |
|-------|--------------|
| 1.5T | 1.35 |
| 3T | 1.65 |
| 7T | 2.1 |

### Blood-Brain Partition Coefficient (λ)

- Grey matter: 0.98 ml/g
- White matter: 0.82 ml/g
- Whole brain average: **0.9 ml/g** (commonly used)

### Labelling Efficiency (α)

| Method | Typical $\alpha$ |
|--------|------------------|
| pCASL | 0.85 |
| PASL | 0.98 |
| CASL | 0.95 |

## Volume Processing

Both ASL quantification functions handle multi-dimensional data:

```python
import numpy as np
from qmri.perfusion import asl

# 3D ASL data
control_3d = np.random.rand(64, 64, 30) * 1000 + 1000
label_3d = control_3d - np.random.rand(64, 64, 30) * 50  # ~5% difference
m0_3d = np.random.rand(64, 64, 30) * 500 + 2000

result = asl.quantify_pcasl(
    control=control_3d,
    label=label_3d,
    m0=m0_3d,
    label_duration=1.8,
    post_label_delay=1.8
)

print(f"CBF map shape: {result.perfusion.shape}")  # (64, 64, 30)
```

## Complete Example: ASL Processing Pipeline

```python
import numpy as np
from qmri.perfusion import asl, gkm

# Simulate a voxel with known perfusion
true_cbf = 60.0  # ml/100g/min
true_att = 1.2   # seconds

# Generate GKM signal
pld = 1.8
tau = 1.8
signal_time = pld + tau

gkm_result = gkm.signal_gkm(
    perfusion_rate=true_cbf,
    transit_time=true_att,
    m0_tissue=1000.0,
    label_duration=tau,
    signal_time=signal_time,
    label_efficiency=0.85,
    partition_coefficient=0.9,
    t1_blood=1.65,
    t1_tissue=1.3,
    label_type="pcasl"
)

# Simulate control and label images
m0 = 1000.0
control_signal = m0  # No labelling effect
label_signal = m0 - gkm_result.delta_m

# Add noise
noise_std = 5.0
control_noisy = control_signal + np.random.normal(0, noise_std)
label_noisy = label_signal + np.random.normal(0, noise_std)
m0_noisy = m0 + np.random.normal(0, noise_std)

# Quantify CBF
asl_result = asl.quantify_pcasl(
    control=np.array([control_noisy]),
    label=np.array([label_noisy]),
    m0=np.array([m0_noisy]),
    label_duration=tau,
    post_label_delay=pld,
    label_efficiency=0.85,
    t1_blood=1.65,
    partition_coefficient=0.9
)

estimated_cbf = asl_result.perfusion[0]
error = 100 * (estimated_cbf - true_cbf) / true_cbf
print(f"True CBF: {true_cbf:.1f} ml/100g/min")
print(f"Estimated CBF: {estimated_cbf:.1f} ml/100g/min")
print(f"Error: {error:+.1f}%")
```

## Best Practices

### Acquisition

1. **Use pCASL** as the default labelling scheme.

2. **Acquire an M0 calibration image** with long TR (>5 s).

3. **Choose PLD based on expected ATT**:
   - Longer PLD for vascular disease or elderly patients
   - Consider multi-PLD for ATT estimation

4. **Use background suppression** to reduce physiological noise.

### Quantification

1. **Use appropriate T1 blood value** for your field strength.

2. **Consider partial volume effects** in grey/white matter boundaries.

3. **Apply motion correction** before averaging control-label pairs.

4. **Mask low M0 regions** to avoid division by small numbers:
   ```python
   mask = m0 > threshold
   result = asl.quantify_pcasl(
       control=control[mask],
       label=label[mask],
       m0=m0[mask],
       ...
   )
   ```

### Quality Control

1. **Inspect control-label subtraction images** for artefacts.

2. **Check for negative CBF values** (may indicate motion or noise).

3. **Verify M0 image quality** (affects all voxels).

4. **Compare with expected CBF ranges**:
   - Grey matter: 50-80 ml/100g/min
   - White matter: 20-30 ml/100g/min

## Typical CBF Values

| Tissue/Region | CBF (ml/100g/min) |
|---------------|-------------------|
| Grey matter | 50-80 |
| White matter | 20-30 |
| Whole brain | 45-55 |
| Cortical grey matter | 60-80 |
| Deep grey matter | 40-60 |

## References

1. Alsop, D.C., et al. "Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications." MRM 73(1):102-116, 2015.

2. Buxton, R.B., et al. "A general kinetic model for quantitative perfusion imaging with arterial spin labeling." MRM 40(3):383-396, 1998.

3. Grade, M., et al. "A neuroradiologist's guide to arterial spin labeling MRI in clinical practice." Neuroradiology 57(12):1181-1202, 2015.
