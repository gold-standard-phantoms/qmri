# Diffusion Equations

This document describes the mathematical models used for diffusion-weighted MRI analysis, including the mono-exponential ADC model and weighted least squares fitting approaches.

## Mono-exponential ADC Model

The standard mono-exponential model describes the signal attenuation in diffusion-weighted imaging as a function of the b-value:

$$
S(b) = S_0 \exp(-b \cdot \text{ADC})
$$

Where:

- $S(b)$ is the measured signal at b-value $b$
- $S_0$ is the signal intensity without diffusion weighting (b = 0)
- $b$ is the diffusion weighting factor (s/mm²)
- $\text{ADC}$ is the Apparent Diffusion Coefficient (mm²/s)

This model assumes:

1. Free (Gaussian) diffusion within the measurement time
2. A single diffusion compartment
3. No flow or perfusion effects

## Log-Linear Fitting Derivation

Taking the natural logarithm of both sides of the mono-exponential model linearises the equation:

$$
\ln(S(b)) = \ln(S_0) - b \cdot \text{ADC}
$$

This can be expressed in linear form as:

$$
y = c + m \cdot x
$$

Where:

- $y = \ln(S(b))$
- $c = \ln(S_0)$
- $m = -\text{ADC}$
- $x = b$

For a set of $n$ measurements at different b-values, the ordinary least squares (OLS) solution minimises:

$$
\chi^2 = \sum_{i=1}^{n} \left( \ln(S_i) - \ln(S_0) + b_i \cdot \text{ADC} \right)^2
$$

The OLS estimates are:

$$
\text{ADC} = \frac{\sum_{i=1}^{n}(b_i - \bar{b})(\ln(S_i) - \overline{\ln(S)})}{\sum_{i=1}^{n}(b_i - \bar{b})^2}
$$

$$
\ln(S_0) = \overline{\ln(S)} + \text{ADC} \cdot \bar{b}
$$

Where $\bar{b}$ and $\overline{\ln(S)}$ are the mean values of $b$ and $\ln(S)$ respectively.

## Weighted Least Squares Formulation

The logarithmic transformation introduces heteroscedasticity (non-uniform variance) in the residuals. For Rician noise with standard deviation $\sigma$ in the original signal domain, the variance in the log domain is approximately:

$$
\text{Var}(\ln(S_i)) \approx \frac{\sigma^2}{S_i^2}
$$

Weighted least squares (WLS) accounts for this by weighting each observation inversely proportional to its variance:

$$
w_i = S_i^2
$$

The WLS cost function becomes:

$$
\chi^2_{\text{WLS}} = \sum_{i=1}^{n} w_i \left( \ln(S_i) - \ln(S_0) + b_i \cdot \text{ADC} \right)^2
$$

In matrix notation, for the linear system $\mathbf{y} = \mathbf{X}\boldsymbol{\beta}$:

$$
\boldsymbol{\beta}_{\text{WLS}} = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}
$$

Where:

- $\mathbf{y} = [\ln(S_1), \ln(S_2), ..., \ln(S_n)]^T$
- $\mathbf{X}$ is the design matrix with columns $[1, -b_i]$
- $\mathbf{W} = \text{diag}(w_1, w_2, ..., w_n)$ is the diagonal weight matrix
- $\boldsymbol{\beta} = [\ln(S_0), \text{ADC}]^T$

## IWLLS Iterative Refinement

Iteratively Weighted Linear Least Squares (IWLLS) refines the WLS estimate by updating the weights based on the fitted signal values rather than the measured values. This approach is more robust to noise, particularly at high b-values where SNR is low.

### Algorithm

1. **Initialisation**: Compute initial estimates using ordinary least squares:

    $$
    \boldsymbol{\beta}^{(0)} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
    $$

2. **Iteration** (for $k = 1, 2, ...$):

    a. Compute predicted signal values from the current estimate:

    $$
    \hat{S}_i^{(k)} = \exp\left(\ln(\hat{S}_0^{(k-1)}) - b_i \cdot \text{ADC}^{(k-1)}\right)
    $$

    b. Update weights using predicted signals:

    $$
    w_i^{(k)} = \left(\hat{S}_i^{(k)}\right)^2
    $$

    c. Solve weighted least squares:

    $$
    \boldsymbol{\beta}^{(k)} = (\mathbf{X}^T \mathbf{W}^{(k)} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W}^{(k)} \mathbf{y}
    $$

3. **Convergence**: Stop when parameters converge:

    $$
    \frac{|\text{ADC}^{(k)} - \text{ADC}^{(k-1)}|}{|\text{ADC}^{(k-1)}|} < \epsilon
    $$

   Typically $\epsilon = 10^{-6}$ or after a maximum number of iterations (e.g., 20).

### Advantages of IWLLS

- **Reduced noise bias**: Using fitted rather than measured signals for weights reduces the influence of noise
- **Better performance at low SNR**: More accurate ADC estimates at high b-values
- **Consistent estimates**: Converges to consistent parameter estimates
- **Closed-form solution**: Each iteration has an analytical solution, making it computationally efficient

## Goodness of Fit

The coefficient of determination $R^2$ quantifies fit quality:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(S_i - \hat{S}_i)^2}{\sum_{i=1}^{n}(S_i - \bar{S})^2}
$$

Where $\hat{S}_i$ are the predicted signal values from the fitted model.

## References

1. Le Bihan D, Breton E, Lallemand D, et al. MR imaging of intravoxel incoherent motions: application to diffusion and perfusion in neurologic disorders. *Radiology*. 1986;161(2):401-407.

2. Veraart J, Sijbers J, Sunaert S, et al. Weighted linear least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. *NeuroImage*. 2013;81:335-346.

3. Salvador R, Peña A, Menon DK, et al. Formal characterization and extension of the linearized diffusion tensor model. *Human Brain Mapping*. 2005;24(2):144-155.
