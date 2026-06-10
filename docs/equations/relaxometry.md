# Relaxometry Equations

This document describes the mathematical models used for T1 and T2 relaxometry in quantitative MRI.

## T1 Relaxometry

### Inversion Recovery

The inversion recovery (IR) sequence is the gold standard for T1 measurement. Following a 180° inversion pulse, the longitudinal magnetisation recovers according to:

$$
S(TI) = \left| M_0 \left( 1 - 2\alpha \exp\left(-\frac{TI}{T_1}\right) + \exp\left(-\frac{TR}{T_1}\right) \right) \right|
$$

Where:

- $S(TI)$ is the measured signal at inversion time $TI$
- $M_0$ is the equilibrium magnetisation (proportional to proton density)
- $\alpha$ is the inversion efficiency (ideally 1.0 for perfect 180° pulse)
- $TI$ is the inversion time (ms)
- $TR$ is the repetition time (ms)
- $T_1$ is the longitudinal relaxation time (ms)

The absolute value accounts for magnitude imaging, which cannot distinguish positive and negative magnetisation.

#### Simplified Form (Long TR)

When $TR \gg T_1$ (typically $TR > 5T_1$), the $\exp(-TR/T_1)$ term becomes negligible:

$$
S(TI) \approx \left| M_0 \left( 1 - 2\alpha \exp\left(-\frac{TI}{T_1}\right) \right) \right|
$$

#### Null Point

The inversion time at which the signal crosses zero (null point) is:

$$
TI_{\text{null}} = T_1 \ln\left(\frac{2\alpha}{1 + \exp(-TR/T_1)}\right)
$$

For long TR: $TI_{\text{null}} \approx T_1 \ln(2\alpha) \approx 0.693 \cdot T_1$ (when $\alpha = 1$).

### Variable TR (VTR) Method

The variable TR method measures T1 by acquiring spoiled gradient echo images at multiple repetition times with fixed flip angle:

$$
S(TR) = M_0 \sin(\theta) \frac{1 - \exp(-TR/T_1)}{1 - \cos(\theta)\exp(-TR/T_1)}
$$

Where:

- $\theta$ is the flip angle

For small flip angles or when $TR \gg T_1$, this simplifies to:

$$
S(TR) \approx M_0 \left( 1 - \exp\left(-\frac{TR}{T_1}\right) \right)
$$

#### Linearisation for VTR

The simplified VTR equation can be linearised as:

$$
\ln(M_0 - S(TR)) = \ln(M_0) - \frac{TR}{T_1}
$$

Or rearranged as:

$$
\frac{S(TR)}{1 - \exp(-TR/T_1)} = M_0
$$

### Variable Flip Angle (VFA) Method

For spoiled gradient echo acquisitions with fixed TR and variable flip angle:

$$
\frac{S(\theta)}{\sin(\theta)} = E_1 \frac{S(\theta)}{\tan(\theta)} + M_0(1 - E_1)
$$

Where $E_1 = \exp(-TR/T_1)$.

This linearisation allows T1 to be extracted from the slope:

$$
T_1 = -\frac{TR}{\ln(E_1)}
$$

## T2 Relaxometry

### Mono-exponential T2 Decay

The transverse magnetisation decays exponentially following excitation:

$$
S(TE) = A \exp\left(-\frac{TE}{T_2}\right) + C
$$

Where:

- $S(TE)$ is the measured signal at echo time $TE$
- $A$ is the signal amplitude at $TE = 0$
- $TE$ is the echo time (ms)
- $T_2$ is the transverse relaxation time (ms)
- $C$ is an optional baseline offset (noise floor)

#### Two-Parameter Model (No Offset)

When the baseline offset is negligible:

$$
S(TE) = S_0 \exp\left(-\frac{TE}{T_2}\right)
$$

This can be linearised by taking the natural logarithm:

$$
\ln(S(TE)) = \ln(S_0) - \frac{TE}{T_2}
$$

### Multi-echo Spin Echo

For a Carr-Purcell-Meiboom-Gill (CPMG) sequence with $n$ echoes:

$$
S(n \cdot \tau) = S_0 \exp\left(-\frac{n \cdot \tau}{T_2}\right)
$$

Where $\tau$ is the inter-echo spacing.

#### Stimulated Echo Contamination

In practice, imperfect refocusing pulses lead to stimulated echo contamination, which can be modelled as:

$$
S(TE) = A \exp\left(-\frac{TE}{T_2}\right) + B \exp\left(-\frac{TE}{T_1}\right)
$$

### T2* Decay

For gradient echo sequences, the observed decay includes contributions from field inhomogeneities:

$$
S(TE) = S_0 \exp\left(-\frac{TE}{T_2^*}\right)
$$

Where:

$$
\frac{1}{T_2^*} = \frac{1}{T_2} + \frac{1}{T_2'}
$$

And $T_2'$ represents the reversible dephasing due to macroscopic field inhomogeneities.

## Fitting Considerations

### Weighted Least Squares

For both T1 and T2 fitting, weighted least squares can improve estimates by accounting for the signal-dependent noise:

$$
w_i = S_i^2
$$

### Non-linear Fitting

When linearisation is not possible (e.g., IR with unknown $\alpha$), non-linear least squares minimises:

$$
\chi^2 = \sum_{i=1}^{n} \left( S_i - S_{\text{model}}(t_i; \boldsymbol{\theta}) \right)^2
$$

Where $\boldsymbol{\theta}$ represents the model parameters.

### Goodness of Fit

The coefficient of determination $R^2$ and root mean square error (RMSE) quantify fit quality:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(S_i - \hat{S}_i)^2}{\sum_{i=1}^{n}(S_i - \bar{S})^2}
$$

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(S_i - \hat{S}_i)^2}
$$

## References

1. Deoni SCL, Peters TM, Rutt BK. High-resolution T1 and T2 mapping of the brain in a clinically acceptable time with DESPOT1 and DESPOT2. *Magnetic Resonance in Medicine*. 2005;53(1):237-241.

2. Barral JK, Gudmundson E, Stikov N, et al. A robust methodology for in vivo T1 mapping. *Magnetic Resonance in Medicine*. 2010;64(4):1057-1067.

3. Poon CS, Henkelman RM. Practical T2 quantitation for clinical applications. *Journal of Magnetic Resonance Imaging*. 1992;2(5):541-553.

4. Look DC, Locker DR. Time saving in measurement of NMR and EPR relaxation times. *Review of Scientific Instruments*. 1970;41(2):250-251.
