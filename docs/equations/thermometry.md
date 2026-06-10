# Thermometry Equations

This document describes the mathematical models used for MR thermometry, focusing on the proton resonance frequency (PRF) shift method for non-invasive temperature measurement.

## Proton Resonance Frequency Shift Method

The PRF shift method exploits the temperature-dependent chemical shift of water protons. As temperature increases, hydrogen bonds weaken, causing water protons to become more shielded and their resonance frequency to decrease.

### Temperature Change Equation

The temperature change is calculated from the phase difference between a baseline and heated image:

$$
\Delta T = \frac{\Delta \phi}{\alpha \cdot \gamma \cdot B_0 \cdot TE}
$$

Where:

- $\Delta T$ is the temperature change (°C or K)
- $\Delta \phi$ is the phase difference (radians)
- $\alpha$ is the PRF temperature coefficient (ppm/°C)
- $\gamma$ is the gyromagnetic ratio (rad/s/T), approximately $2.675 \times 10^8$ rad/s/T for protons
- $B_0$ is the main magnetic field strength (T)
- $TE$ is the echo time (s)

### PRF Temperature Coefficient

The PRF temperature coefficient $\alpha$ is approximately:

$$
\alpha \approx -0.01 \text{ ppm/°C}
$$

This value is relatively tissue-independent for aqueous tissues but does vary slightly:

- Pure water: $-0.0100$ ppm/°C
- Muscle: $-0.0097$ to $-0.0100$ ppm/°C
- Liver: $-0.0090$ to $-0.0100$ ppm/°C
- Fat: ~0 ppm/°C (fat is insensitive to PRF shift)

The negative sign indicates that the resonance frequency decreases with increasing temperature.

### Phase Accumulation

The phase of a gradient echo signal at time $TE$ is:

$$
\phi = \gamma \cdot B_0 \cdot (1 + \delta(T)) \cdot TE + \phi_0
$$

Where:

- $\delta(T)$ is the temperature-dependent chemical shift (ppm)
- $\phi_0$ includes all temperature-independent phase contributions

The phase difference between temperatures $T$ and $T_0$ is:

$$
\Delta \phi = \phi(T) - \phi(T_0) = \gamma \cdot B_0 \cdot (\delta(T) - \delta(T_0)) \cdot TE
$$

Since $\delta(T) - \delta(T_0) = \alpha \cdot (T - T_0) = \alpha \cdot \Delta T$:

$$
\Delta \phi = \gamma \cdot B_0 \cdot \alpha \cdot \Delta T \cdot TE
$$

## Baseline Subtraction

The PRF method requires a baseline (reference) phase image acquired at a known temperature:

$$
\Delta \phi = \phi_{\text{heated}} - \phi_{\text{baseline}}
$$

### Baseline Drift Correction

Field drift during the measurement can cause errors. Reference regions with known constant temperature can be used for correction:

$$
\Delta \phi_{\text{corrected}} = \Delta \phi_{\text{measured}} - \Delta \phi_{\text{drift}}
$$

Where $\Delta \phi_{\text{drift}}$ is estimated from reference regions (e.g., oil phantoms, unheated tissue).

### Multi-baseline Approaches

To account for motion, multiple baseline images can be acquired and the best match selected:

$$
\Delta \phi = \phi_{\text{heated}} - \phi_{\text{baseline}}^{(k)}
$$

Where $k$ is selected to minimise the phase difference in non-heated regions.

## Temperature Sensitivity

The temperature sensitivity depends on field strength and echo time:

$$
\frac{\partial \phi}{\partial T} = \alpha \cdot \gamma \cdot B_0 \cdot TE
$$

### At 3T with TE = 10 ms:

$$
\frac{\partial \phi}{\partial T} = (-0.01 \times 10^{-6}) \cdot (2.675 \times 10^8) \cdot 3 \cdot 0.01 = -0.080 \text{ rad/°C}
$$

Or approximately **4.6°/°C** (degrees of phase per degree Celsius).

### Optimal Echo Time

For gradient echo sequences, the optimal TE balances sensitivity and SNR:

$$
TE_{\text{opt}} \approx T_2^*
$$

Longer TE provides greater temperature sensitivity but lower SNR due to $T_2^*$ decay.

## Temperature Uncertainty

The temperature uncertainty depends on the phase noise, which is related to the signal-to-noise ratio:

$$
\sigma_T = \frac{\sigma_\phi}{|\alpha| \cdot \gamma \cdot B_0 \cdot TE}
$$

Where the phase noise standard deviation is:

$$
\sigma_\phi \approx \frac{1}{\text{SNR}}
$$

Therefore:

$$
\sigma_T \approx \frac{1}{|\alpha| \cdot \gamma \cdot B_0 \cdot TE \cdot \text{SNR}}
$$

## Multi-echo Thermometry

Using multiple echoes can improve precision by weighted averaging:

$$
\Delta T = \frac{\sum_{j=1}^{n} w_j \cdot \Delta T_j}{\sum_{j=1}^{n} w_j}
$$

Where the optimal weights are:

$$
w_j = \frac{TE_j^2}{\sigma_j^2}
$$

Or using a linear fit through the phase evolution:

$$
\Delta \phi = m \cdot TE
$$

Where $m = \alpha \cdot \gamma \cdot B_0 \cdot \Delta T$.

## Fat-Referenced Thermometry

Fat protons do not exhibit a significant PRF shift with temperature due to the absence of hydrogen bonding. This can be exploited for drift correction:

$$
\Delta T = \frac{\phi_{\text{water}} - \phi_{\text{water,baseline}} - (\phi_{\text{fat}} - \phi_{\text{fat,baseline}})}{|\alpha| \cdot \gamma \cdot B_0 \cdot TE}
$$

## Absolute Temperature Estimation

The PRF method measures temperature changes, not absolute temperature. Absolute temperature can be estimated using:

$$
T_{\text{absolute}} = T_{\text{baseline}} + \Delta T
$$

Alternative methods for absolute temperature include:

- T1-based thermometry (T1 varies with temperature)
- Diffusion-based thermometry (ADC varies with temperature)

### T1 Temperature Dependence

$$
T_1(T) = T_1(T_0) \cdot \exp\left(\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_0}\right)\right)
$$

Where $E_a$ is the activation energy and $R$ is the gas constant.

## Practical Considerations

### Phase Wrapping

Phase values are inherently wrapped to $[-\pi, \pi]$. For large temperature changes:

$$
\Delta \phi_{\text{unwrapped}} = \text{unwrap}(\Delta \phi_{\text{wrapped}})
$$

Phase unwrapping algorithms are required when $|\Delta \phi| > \pi$.

### Motion Correction

Patient motion causes phase changes unrelated to temperature. Strategies include:

1. Navigator echoes for motion tracking
2. Referenceless methods using polynomial fitting
3. Multi-baseline libraries

### Field Inhomogeneity

Static field inhomogeneities $\Delta B_0$ contribute to phase:

$$
\phi_{\text{total}} = \phi_{\text{temperature}} + \gamma \cdot \Delta B_0 \cdot TE
$$

These should remain constant if baseline subtraction is performed correctly.

## Multi-Echo Dual-Resonance Thermometry

The multi-echo dual-resonance method is designed for **ethylene glycol phantom thermometry**, where the sample contains two chemical species with different resonance frequencies (CH₂ and OH groups).

### Dual-Resonance Signal Model

The multi-echo magnitude signal is modelled as:

$$
S(t) = \sqrt{A_1^2 e^{-2R_{2,1}^* t} + A_2^2 e^{-2R_{2,2}^* t} + 2 A_1 A_2 e^{-(R_{2,1}^* + R_{2,2}^*) t} \cos(2\pi \Delta f_{12} t + \Delta\phi_{12})}
$$

Where:

- $A_1, A_2$ are the amplitudes of the two resonance peaks
- $R_{2,1}^*, R_{2,2}^*$ are the R2* relaxation rates (1/s)
- $\Delta f_{12}$ is the frequency difference between the peaks (Hz)
- $\Delta\phi_{12}$ is the initial phase difference (radians)
- $t$ is the echo time (s)

The frequency difference $\Delta f_{12}$ is related to temperature through a substance-specific calibration.

### Temperature-Frequency Relationship for Ethylene Glycol

For ethylene glycol, the temperature-frequency relationship is:

$$
T[°C] = 193.35 - \frac{1.02 \times 10^8 \cdot |\Delta f_{12}|}{\gamma \cdot B_0}
$$

Where:

- $T$ is the temperature in degrees Celsius
- $\gamma$ is the gyromagnetic ratio for protons (42.58 MHz/T)
- $B_0$ is the main magnetic field strength (T)

The inverse relationship gives frequency from temperature:

$$
\Delta f_{12} = \frac{(193.35 - T) \cdot \gamma \cdot B_0}{1.02 \times 10^8}
$$

### Temperature Uncertainty

The uncertainty in temperature from frequency uncertainty is:

$$
u(T) = \frac{1.02 \times 10^8 \cdot u(\Delta f_{12})}{\gamma \cdot B_0}
$$

Where $u(\Delta f_{12})$ is the uncertainty in the fitted frequency difference.

### Fitting Methodology

The signal model is fitted to multi-echo magnitude data using nonlinear least squares with bounds:

1. Initial parameter estimates are based on the signal amplitude
2. The curve fitting optimises all six parameters simultaneously
3. The fitted frequency difference is converted to temperature
4. Fit quality is assessed using the coefficient of determination (R²)

For regionwise analysis with multiple voxels, bootstrap resampling can estimate parameter uncertainty by:

1. Repeatedly sampling voxels within a region with replacement
2. Fitting the mean signal of each sample
3. Computing the standard deviation of fitted temperatures

### Comparison: PRF vs Multi-Echo Methods

| Aspect | PRF Shift | Multi-Echo Dual-Resonance |
|--------|-----------|---------------------------|
| **Signal type** | Phase difference images | Multi-echo magnitude images |
| **Sample type** | Aqueous tissue (water-based) | Ethylene glycol phantoms |
| **Physics** | Temperature-dependent chemical shift | Two-component resonance model |
| **Output** | Relative temperature change | Absolute temperature |
| **Primary use** | In-vivo thermal therapy monitoring | Phantom calibration/validation |
| **Computation** | Simple division | Nonlinear least-squares fitting |

## References

1. Rieke V, Butts Pauly K. MR thermometry. *Journal of Magnetic Resonance Imaging*. 2008;27(2):376-390.

2. Ishihara Y, Calderon A, Watanabe H, et al. A precise and fast temperature mapping using water proton chemical shift. *Magnetic Resonance in Medicine*. 1995;34(6):814-823.

3. De Poorter J, De Wagter C, De Deene Y, et al. Noninvasive MRI thermometry with the proton resonance frequency (PRF) method: in vivo results in human muscle. *Magnetic Resonance in Medicine*. 1995;33(1):74-81.

4. Quesson B, de Zwart JA, Moonen CT. Magnetic resonance temperature imaging for guidance of thermotherapy. *Journal of Magnetic Resonance Imaging*. 2000;12(4):525-533.

5. Denis de Senneville B, Quesson B, Moonen CT. Magnetic resonance temperature imaging. *International Journal of Hyperthermia*. 2005;21(6):515-531.

6. Sprinkhuizen SM, Bakker CJG, Bartels LW. Absolute MR thermometry using time-domain analysis of multi-gradient-echo magnitude images. *Magnetic Resonance in Medicine*. 2010;64:239-248.

7. Raiford DS, Fisk CL, Becker ED. Calibration of methanol and ethylene glycol nuclear magnetic resonance thermometers. *Analytical Chemistry*. 1979;51(12):2050-2051.
