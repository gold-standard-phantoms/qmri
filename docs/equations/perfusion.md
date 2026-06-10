# Perfusion Equations

This document describes the mathematical models used for perfusion quantification in arterial spin labelling (ASL) MRI.

## Arterial Spin Labelling Overview

ASL uses magnetically labelled arterial blood water as an endogenous tracer to measure cerebral blood flow (CBF). The difference between label and control images is proportional to blood flow:

$$
\Delta M = M_{\text{control}} - M_{\text{label}}
$$

## ASL White Paper Quantification Equation

The recommended quantification approach from the ASL White Paper consensus statement uses the following equation for pulsed or pseudo-continuous ASL with a single post-label delay (PLD):

$$
\text{CBF} = \frac{6000 \cdot \lambda \cdot \Delta M \cdot \exp(PLD/T_{1,\text{blood}})}{2 \cdot \alpha \cdot T_{1,\text{blood}} \cdot M_0 \cdot (1 - \exp(-\tau/T_{1,\text{blood}}))}
$$

Where:

- $\text{CBF}$ is the cerebral blood flow (ml/100g/min)
- $\lambda$ is the blood-brain partition coefficient (ml/g), typically 0.9 ml/g
- $\Delta M$ is the difference signal between control and label images
- $PLD$ is the post-label delay (ms)
- $T_{1,\text{blood}}$ is the longitudinal relaxation time of arterial blood (ms), typically 1650 ms at 3T
- $\alpha$ is the labelling efficiency, typically 0.85 for PCASL, 0.98 for PASL
- $\tau$ is the label duration (ms), typically 1800 ms for PCASL
- $M_0$ is the equilibrium magnetisation of brain tissue
- The factor of 6000 converts from ml/g/s to ml/100g/min

### Simplified Form (Long Label Duration)

When $\tau \gg T_{1,\text{blood}}$:

$$
\text{CBF} = \frac{6000 \cdot \lambda \cdot \Delta M \cdot \exp(PLD/T_{1,\text{blood}})}{2 \cdot \alpha \cdot T_{1,\text{blood}} \cdot M_0}
$$

### Background Suppression Correction

When background suppression is used:

$$
\text{CBF}_{\text{corrected}} = \frac{\text{CBF}}{\alpha_{\text{BS}}}
$$

Where $\alpha_{\text{BS}}$ is the background suppression efficiency (product of inversion efficiencies).

## General Kinetic Model

The General Kinetic Model (GKM) provides a more complete description of the ASL signal, accounting for the temporal dynamics of label arrival and clearance.

### Buxton Model

For pseudo-continuous ASL with instantaneous exchange:

$$
\Delta M(t) = \begin{cases}
0 & t < \Delta t \\
2 \cdot M_{0,\text{blood}} \cdot f \cdot \alpha \cdot T_{1,\text{app}} \cdot \exp(-\Delta t/T_{1,\text{blood}}) \cdot q_p(t) & \Delta t \leq t < \tau + \Delta t \\
2 \cdot M_{0,\text{blood}} \cdot f \cdot \alpha \cdot T_{1,\text{app}} \cdot \exp(-\Delta t/T_{1,\text{blood}}) \cdot q_p(t) \cdot \exp(-(t-\tau-\Delta t)/T_{1,\text{app}}) & t \geq \tau + \Delta t
\end{cases}
$$

Where:

- $f$ is the blood flow (ml/g/s)
- $\Delta t$ is the arterial transit time (ATT) (s)
- $T_{1,\text{app}}$ is the apparent T1 in tissue
- $q_p(t)$ is the delivery function

### Apparent T1

The apparent longitudinal relaxation time accounts for both tissue T1 and the clearance of labelled water:

$$
\frac{1}{T_{1,\text{app}}} = \frac{1}{T_{1,\text{tissue}}} + \frac{f}{\lambda}
$$

### Delivery Function

The delivery function $q_p(t)$ describes the accumulation of labelled blood water:

$$
q_p(t) = 1 - \exp\left(-\frac{t - \Delta t}{T_{1,\text{app}}}\right)
$$

For $\Delta t \leq t < \tau + \Delta t$.

## Multi-PLD ASL

When acquiring data at multiple post-label delays, both CBF and ATT can be estimated by fitting the signal model.

### Signal Model

$$
\Delta M(PLD) = \begin{cases}
0 & PLD < ATT \\
2 \cdot M_0 \cdot f \cdot \alpha \cdot T_{1,\text{blood}} \cdot \exp(-ATT/T_{1,\text{blood}}) \cdot (1 - \exp(-(PLD-ATT)/T_{1,\text{app}})) & ATT \leq PLD < ATT + \tau \\
\text{decay term} & PLD \geq ATT + \tau
\end{cases}
$$

### Arterial Transit Time Estimation

The ATT can be estimated from multi-PLD data using:

1. **Non-linear fitting**: Fit the full kinetic model to the signal-time curve
2. **Weighted delay**: Use the weighted average of delays

$$
ATT \approx \frac{\sum_i PLD_i \cdot \Delta M_i}{\sum_i \Delta M_i}
$$

## Parameter Definitions

| Parameter | Symbol | Typical Value (3T) | Units |
|-----------|--------|-------------------|-------|
| Blood-brain partition coefficient | $\lambda$ | 0.9 | ml/g |
| T1 of arterial blood | $T_{1,\text{blood}}$ | 1650 | ms |
| T1 of grey matter | $T_{1,\text{GM}}$ | 1300 | ms |
| T1 of white matter | $T_{1,\text{WM}}$ | 830 | ms |
| Labelling efficiency (PCASL) | $\alpha$ | 0.85 | - |
| Labelling efficiency (PASL) | $\alpha$ | 0.98 | - |
| Label duration (PCASL) | $\tau$ | 1800 | ms |
| Post-label delay | $PLD$ | 1800-2000 | ms |
| Arterial transit time (GM) | $ATT$ | 500-1500 | ms |
| Normal grey matter CBF | $f$ | 50-80 | ml/100g/min |
| Normal white matter CBF | $f$ | 20-30 | ml/100g/min |

## Partial Volume Correction

For partial volume effects between grey matter (GM), white matter (WM), and cerebrospinal fluid (CSF):

$$
\Delta M = f_{\text{GM}} \cdot \Delta M_{\text{GM}} + f_{\text{WM}} \cdot \Delta M_{\text{WM}}
$$

Where $f_{\text{GM}}$ and $f_{\text{WM}}$ are the volume fractions.

Linear regression can be used to separate tissue contributions:

$$
\text{CBF}_{\text{GM}} = \frac{\text{CBF}_{\text{measured}} - f_{\text{WM}} \cdot \text{CBF}_{\text{WM}}}{f_{\text{GM}}}
$$

## References

1. Alsop DC, Detre JA, Golay X, et al. Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications: A consensus of the ISMRM perfusion study group and the European consortium for ASL in dementia. *Magnetic Resonance in Medicine*. 2015;73(1):102-116.

2. Buxton RB, Frank LR, Wong EC, et al. A general kinetic model for quantitative perfusion imaging with arterial spin labeling. *Magnetic Resonance in Medicine*. 1998;40(3):383-396.

3. Chappell MA, MacIntosh BJ, Donahue MJ, et al. Separation of macrovascular signal in multi-inversion time arterial spin labelling MRI. *Magnetic Resonance in Medicine*. 2010;63(5):1357-1365.

4. Petersen ET, Zimine I, Ho YC, Golay X. Non-invasive measurement of perfusion: a critical review of arterial spin labelling techniques. *British Journal of Radiology*. 2006;79(944):688-701.
