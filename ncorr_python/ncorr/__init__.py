"""
Ncorr 2D - Open-source Digital Image Correlation (DIC) software in Python.

This is a Python translation of the original MATLAB Ncorr software.

Reference:
    Ncorr: open-source 2D digital image correlation matlab software
    J Blaber, B Adair, A Antoniou
    Experimental Mechanics 55 (6), 1105-1122
"""

from .core.status import Status
from .core.image import NcorrImage
from .core.roi import NcorrROI, Region, Boundary
from .core.dic_parameters import DICParameters
from .algorithms.bspline import BSplineInterpolator
from .algorithms.dic import DICAnalysis
from .algorithms.strain import StrainCalculator, StrainResult
from .algorithms.regions import RegionProcessor
from .main import Ncorr, AnalysisResults

__version__ = "2.0.0"
__author__ = "Ncorr Python Translation"

__all__ = [
    "Ncorr",
    "AnalysisResults",
    "Status",
    "NcorrImage",
    "NcorrROI",
    "Region",
    "Boundary",
    "DICParameters",
    "BSplineInterpolator",
    "DICAnalysis",
    "StrainCalculator",
    "StrainResult",
    "RegionProcessor",
]
