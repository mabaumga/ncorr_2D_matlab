"""
DIC Parameters configuration.

Stores all parameters used for DIC analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StepAnalysisType(str, Enum):
    """Step analysis types."""

    REGULAR = "regular"
    BACKWARD = "backward"


@dataclass
class StepAnalysis:
    """
    Step analysis configuration for high-strain cases.

    Attributes:
        enabled: Whether step analysis is enabled
        type: Type of step analysis (regular or backward)
        auto: Use automatic threshold detection
        step: Number of images per step
    """

    enabled: bool = False
    type: StepAnalysisType = StepAnalysisType.REGULAR
    auto: bool = True
    step: int = 1


@dataclass
class DICParameters:
    """
    DIC analysis parameters.

    Attributes:
        radius: Subset radius in pixels (typical: 30-50)
        spacing: Spacing between subset centers (typical: 5-10)
        cutoff_diffnorm: Convergence tolerance for optimization (typical: 1e-2)
        cutoff_iteration: Maximum iterations for optimization (typical: 20)
        total_threads: Number of parallel threads
        step_analysis: Step analysis configuration
        subset_trunc: Enable subset truncation for discontinuities
        pix_to_units: Pixel to physical unit conversion factor
        units: Physical unit string (e.g., 'mm', 'in')
        lens_coef: Radial lens distortion coefficient
    """

    radius: int = 30
    spacing: int = 5
    cutoff_diffnorm: float = 1e-2
    cutoff_iteration: int = 20
    total_threads: int = 1
    step_analysis: StepAnalysis = field(default_factory=StepAnalysis)
    subset_trunc: bool = False
    pix_to_units: float = 1.0
    units: str = "pixels"
    lens_coef: float = 0.0

    def validate(self) -> bool:
        """
        Validate parameters are within acceptable ranges.

        Returns:
            True if parameters are valid, raises ValueError otherwise
        """
        if not (10 <= self.radius <= 200):
            raise ValueError(f"Radius must be between 10 and 200, got {self.radius}")

        if not (0 <= self.spacing <= 80):
            raise ValueError(f"Spacing must be between 0 and 80, got {self.spacing}")

        if not (1e-8 <= self.cutoff_diffnorm <= 1):
            raise ValueError(
                f"cutoff_diffnorm must be between 1e-8 and 1, got {self.cutoff_diffnorm}"
            )

        if not (5 <= self.cutoff_iteration <= 100):
            raise ValueError(
                f"cutoff_iteration must be between 5 and 100, got {self.cutoff_iteration}"
            )

        if self.total_threads < 1:
            raise ValueError(f"total_threads must be >= 1, got {self.total_threads}")

        return True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "radius": self.radius,
            "spacing": self.spacing,
            "cutoff_diffnorm": self.cutoff_diffnorm,
            "cutoff_iteration": self.cutoff_iteration,
            "total_threads": self.total_threads,
            "step_analysis": {
                "enabled": self.step_analysis.enabled,
                "type": self.step_analysis.type.value,
                "auto": self.step_analysis.auto,
                "step": self.step_analysis.step,
            },
            "subset_trunc": self.subset_trunc,
            "pix_to_units": self.pix_to_units,
            "units": self.units,
            "lens_coef": self.lens_coef,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DICParameters":
        """Create from dictionary."""
        step_analysis_data = d.get("step_analysis", {})
        step_analysis = StepAnalysis(
            enabled=step_analysis_data.get("enabled", False),
            type=StepAnalysisType(step_analysis_data.get("type", "regular")),
            auto=step_analysis_data.get("auto", True),
            step=step_analysis_data.get("step", 1),
        )

        return cls(
            radius=d.get("radius", 30),
            spacing=d.get("spacing", 5),
            cutoff_diffnorm=d.get("cutoff_diffnorm", 1e-2),
            cutoff_iteration=d.get("cutoff_iteration", 20),
            total_threads=d.get("total_threads", 1),
            step_analysis=step_analysis,
            subset_trunc=d.get("subset_trunc", False),
            pix_to_units=d.get("pix_to_units", 1.0),
            units=d.get("units", "pixels"),
            lens_coef=d.get("lens_coef", 0.0),
        )
