# qmri

Pure MRI signal models, fitting algorithms, and error propagation for quantitative MRI.

## Installation

```bash
pip install qmri
```

## Quick Start

```python
import numpy as np
from qmri.diffusion import adc

b_values = np.array([0, 500, 1000, 2000])
signal = np.array([1000, 606, 368, 135])
result = adc.fit(signal, b_values, method="iwlls")

print(f"ADC: {result.adc:.2e} mm²/s")
print(f"R²: {result.r_squared:.4f}")
```

## Documentation

See the full documentation at [qmri.readthedocs.io](https://qmri.readthedocs.io).

## License

MIT
