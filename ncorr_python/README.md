# Ncorr 2D - Python

A Python translation of the Ncorr 2D Digital Image Correlation (DIC) software.

## Overview

Ncorr is an open-source 2D Digital Image Correlation (DIC) software that enables measurement of full-field deformation and strain from digital images. This Python version is a complete translation of the original MATLAB implementation.

**Reference:**
> Ncorr: open-source 2D digital image correlation matlab software
> J Blaber, B Adair, A Antoniou
> Experimental Mechanics 55 (6), 1105-1122

## Features

- **Digital Image Correlation (DIC)**: Measure displacements between reference and deformed images
- **Strain Calculation**: Compute Green-Lagrange and Eulerian-Almansi strains
- **Region of Interest (ROI)**: Define analysis regions via masks
- **B-spline Interpolation**: Subpixel accuracy using biquintic B-splines
- **Multi-image Analysis**: Process image sequences with step analysis
- **Performance**: Optimized with NumPy and Numba JIT compilation

## Installation

```bash
# From source
cd ncorr_python
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from ncorr import Ncorr, DICParameters

# Create Ncorr instance
ncorr = Ncorr()

# Set images
ncorr.set_reference("reference.tif")
ncorr.set_current(["deformed_001.tif", "deformed_002.tif"])

# Set ROI from mask
ncorr.set_roi_from_mask(mask_array)

# Configure parameters
params = DICParameters(
    radius=30,        # Subset radius (pixels)
    spacing=5,        # Grid spacing
    cutoff_diffnorm=1e-3,
    cutoff_iteration=30,
)
ncorr.set_parameters(params)

# Run analysis
results = ncorr.run_analysis()

# Access results
for i, disp in enumerate(results.displacements):
    print(f"Image {i}: u_mean={disp.u[disp.roi].mean():.3f}")
```

## Project Structure

```
ncorr_python/
├── ncorr/
│   ├── __init__.py
│   ├── main.py                 # Main API
│   ├── core/
│   │   ├── status.py          # Status enumeration
│   │   ├── image.py           # NcorrImage class
│   │   ├── roi.py             # NcorrROI class
│   │   └── dic_parameters.py  # DIC parameters
│   ├── algorithms/
│   │   ├── bspline.py         # B-spline interpolation
│   │   ├── dic.py             # DIC analysis
│   │   ├── strain.py          # Strain calculation
│   │   ├── seeds.py           # Seed calculation
│   │   └── regions.py         # Region processing
│   └── utils/
│       ├── image_loader.py    # Image loading
│       ├── validation.py      # Parameter validation
│       └── colormaps.py       # Visualization
├── tests/
│   ├── test_core.py
│   ├── test_algorithms.py
│   ├── test_main.py
│   └── test_utils.py
└── pyproject.toml
```

## Core Classes

### NcorrImage
Represents an image with B-spline coefficients for interpolation:

```python
from ncorr.core.image import NcorrImage

# Load from file
img = NcorrImage.from_file("image.tif")

# Load from array
img = NcorrImage.from_array(numpy_array)

# Access data
gs = img.get_gs()       # Grayscale values
bcoef = img.get_bcoef() # B-spline coefficients
```

### NcorrROI
Defines the region of interest for analysis:

```python
from ncorr.core.roi import NcorrROI

roi = NcorrROI()
roi.set_roi("load", {"mask": binary_mask, "cutoff": 20})

# Access regions
for region in roi.regions:
    print(f"Region: {region.totalpoints} points")
```

### DICParameters
Configuration for DIC analysis:

```python
from ncorr.core.dic_parameters import DICParameters

params = DICParameters(
    radius=30,              # Subset radius
    spacing=5,              # Grid spacing
    cutoff_diffnorm=1e-3,   # Convergence tolerance
    cutoff_iteration=30,    # Max iterations
    subset_trunc=False,     # Subset truncation
)
```

## Algorithms

### DIC Analysis
The core DIC algorithm uses Inverse Compositional Gauss-Newton (IC-GN) optimization:

```python
from ncorr.algorithms.dic import DICAnalysis, SeedInfo

# Create analysis
dic = DICAnalysis(params)

# Define seeds (initial guesses)
seeds = [SeedInfo(x=100, y=100, u=0, v=0, region_idx=0, valid=True)]

# Run analysis
results = dic.analyze(ref_img, cur_imgs, roi, seeds)
```

### Strain Calculation
Calculate strains from displacement fields:

```python
from ncorr.algorithms.strain import StrainCalculator

calc = StrainCalculator(strain_radius=5)

# Green-Lagrange strain (reference configuration)
strain_ref = calc.calculate_green_lagrange(u, v, roi, spacing)

# Eulerian-Almansi strain (current configuration)
strain_cur = calc.calculate_eulerian_almansi(u, v, roi, spacing)

# Principal strains
e1, e2, theta = StrainCalculator.calculate_principal_strains(
    strain_ref.exx, strain_ref.exy, strain_ref.eyy
)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ncorr

# Run specific tests
pytest tests/test_core.py -v
```

## Dependencies

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7
- Pillow >= 9.0
- Numba >= 0.55
- OpenCV >= 4.5
- Matplotlib >= 3.5

## Differences from MATLAB Version

1. **No MEX files**: All C++ algorithms are reimplemented in Python with Numba JIT
2. **No GUI by default**: Programmatic API only (GUI can be added with PyQt6)
3. **NumPy arrays**: Uses NumPy instead of MATLAB matrices
4. **Python idioms**: Uses dataclasses, type hints, and Pythonic patterns

## License

This is a translation of the open-source Ncorr software. See the original project for license information.

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.
