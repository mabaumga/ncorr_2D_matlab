"""
Main Ncorr application module.

Provides high-level API for DIC analysis workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable, Union, Dict, Any
import json

import numpy as np
from numpy.typing import NDArray

from .core.image import NcorrImage
from .core.roi import NcorrROI
from .core.dic_parameters import DICParameters
from .core.status import Status
from .algorithms.dic import DICAnalysis, DICResult, SeedInfo
from .algorithms.strain import StrainCalculator, StrainResult
from .algorithms.seeds import SeedCalculator


@dataclass
class AnalysisResults:
    """
    Complete DIC analysis results.

    Attributes:
        displacements: List of displacement results per image
        strains_ref: Strains in reference configuration
        strains_cur: Strains in current configuration
        parameters: Analysis parameters used
    """

    displacements: List[DICResult] = field(default_factory=list)
    strains_ref: List[StrainResult] = field(default_factory=list)
    strains_cur: List[StrainResult] = field(default_factory=list)
    parameters: Optional[DICParameters] = None

    def save(self, filepath: Union[str, Path]) -> None:
        """Save results to file."""
        filepath = Path(filepath)

        data = {
            "parameters": self.parameters.to_dict() if self.parameters else None,
            "num_images": len(self.displacements),
        }

        # Save as NPZ for arrays
        arrays = {}
        for i, disp in enumerate(self.displacements):
            arrays[f"u_{i}"] = disp.u
            arrays[f"v_{i}"] = disp.v
            arrays[f"corrcoef_{i}"] = disp.corrcoef
            arrays[f"roi_{i}"] = disp.roi

        for i, strain in enumerate(self.strains_ref):
            arrays[f"exx_ref_{i}"] = strain.exx
            arrays[f"exy_ref_{i}"] = strain.exy
            arrays[f"eyy_ref_{i}"] = strain.eyy

        for i, strain in enumerate(self.strains_cur):
            arrays[f"exx_cur_{i}"] = strain.exx
            arrays[f"exy_cur_{i}"] = strain.exy
            arrays[f"eyy_cur_{i}"] = strain.eyy

        np.savez(filepath.with_suffix(".npz"), **arrays)

        # Save metadata as JSON
        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "AnalysisResults":
        """Load results from file."""
        filepath = Path(filepath)

        # Load metadata
        with open(filepath.with_suffix(".json")) as f:
            data = json.load(f)

        # Load arrays
        arrays = np.load(filepath.with_suffix(".npz"))

        results = cls()
        results.parameters = DICParameters.from_dict(data["parameters"]) if data["parameters"] else None

        # Reconstruct results
        for i in range(data["num_images"]):
            disp = DICResult(
                u=arrays[f"u_{i}"],
                v=arrays[f"v_{i}"],
                corrcoef=arrays[f"corrcoef_{i}"],
                roi=arrays[f"roi_{i}"],
            )
            results.displacements.append(disp)

            if f"exx_ref_{i}" in arrays:
                strain_ref = StrainResult(
                    exx=arrays[f"exx_ref_{i}"],
                    exy=arrays[f"exy_ref_{i}"],
                    eyy=arrays[f"eyy_ref_{i}"],
                    roi=disp.roi,
                )
                results.strains_ref.append(strain_ref)

            if f"exx_cur_{i}" in arrays:
                strain_cur = StrainResult(
                    exx=arrays[f"exx_cur_{i}"],
                    exy=arrays[f"exy_cur_{i}"],
                    eyy=arrays[f"eyy_cur_{i}"],
                    roi=disp.roi,
                )
                results.strains_cur.append(strain_cur)

        return results


class Ncorr:
    """
    Main Ncorr application class.

    Provides high-level API for DIC analysis workflow:
    1. Load reference and current images
    2. Define region of interest (ROI)
    3. Set analysis parameters
    4. Run DIC analysis
    5. Calculate strains
    6. Export results

    Example:
        >>> ncorr = Ncorr()
        >>> ncorr.set_reference("ref.tif")
        >>> ncorr.set_current(["cur_001.tif", "cur_002.tif"])
        >>> ncorr.set_roi_from_image(ref_mask)
        >>> ncorr.set_parameters(DICParameters(radius=30, spacing=5))
        >>> results = ncorr.run_analysis()
    """

    def __init__(self):
        """Initialize Ncorr application."""
        self._ref_img: Optional[NcorrImage] = None
        self._cur_imgs: List[NcorrImage] = []
        self._roi_ref: Optional[NcorrROI] = None
        self._roi_cur: List[NcorrROI] = []
        self._params: DICParameters = DICParameters()
        self._seeds: List[SeedInfo] = []
        self._results: Optional[AnalysisResults] = None
        self._progress_callback: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """
        Set callback for progress updates.

        Args:
            callback: Function(progress: float, message: str)
        """
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str = "") -> None:
        """Report progress to callback."""
        if self._progress_callback:
            self._progress_callback(progress, message)

    # Image management

    def set_reference(
        self,
        source: Union[str, Path, NDArray, NcorrImage],
        lazy: bool = False,
    ) -> Status:
        """
        Set reference image.

        Args:
            source: Image file path, numpy array, or NcorrImage
            lazy: Use lazy loading for file

        Returns:
            Status
        """
        try:
            if isinstance(source, NcorrImage):
                self._ref_img = source
            elif isinstance(source, np.ndarray):
                self._ref_img = NcorrImage.from_array(source)
            else:
                if lazy:
                    self._ref_img = NcorrImage()
                    path = Path(source)
                    self._ref_img.set_image(
                        "lazy",
                        {"name": path.name, "path": str(path.parent)}
                    )
                else:
                    self._ref_img = NcorrImage.from_file(source)

            # Clear dependent data
            self._roi_ref = None
            self._seeds = []
            self._results = None

            return Status.SUCCESS

        except Exception as e:
            print(f"Error setting reference: {e}")
            return Status.FAILED

    def set_current(
        self,
        sources: Union[str, Path, NDArray, NcorrImage, List],
        lazy: bool = False,
    ) -> Status:
        """
        Set current image(s).

        Args:
            sources: Single or list of image sources
            lazy: Use lazy loading

        Returns:
            Status
        """
        try:
            if not isinstance(sources, list):
                sources = [sources]

            self._cur_imgs = []

            for source in sources:
                if isinstance(source, NcorrImage):
                    img = source
                elif isinstance(source, np.ndarray):
                    img = NcorrImage.from_array(source)
                else:
                    if lazy:
                        img = NcorrImage()
                        path = Path(source)
                        img.set_image(
                            "lazy",
                            {"name": path.name, "path": str(path.parent)}
                        )
                    else:
                        img = NcorrImage.from_file(source)

                self._cur_imgs.append(img)

            # Clear dependent data
            self._seeds = []
            self._results = None

            return Status.SUCCESS

        except Exception as e:
            print(f"Error setting current images: {e}")
            return Status.FAILED

    # ROI management

    def set_roi_from_mask(
        self,
        mask: NDArray[np.bool_],
        min_region_size: int = 20,
    ) -> Status:
        """
        Set ROI from binary mask.

        Args:
            mask: Binary mask array
            min_region_size: Minimum pixels for valid region

        Returns:
            Status
        """
        try:
            self._roi_ref = NcorrROI()
            self._roi_ref.set_roi(
                "load",
                {"mask": mask, "cutoff": min_region_size}
            )

            # Clear dependent data
            self._seeds = []
            self._results = None

            return Status.SUCCESS

        except Exception as e:
            print(f"Error setting ROI: {e}")
            return Status.FAILED

    def set_roi_from_image(
        self,
        mask_image: Union[str, Path, NDArray],
        threshold: float = 0.5,
    ) -> Status:
        """
        Set ROI from mask image.

        Args:
            mask_image: Path to mask image or array
            threshold: Threshold for binarization

        Returns:
            Status
        """
        try:
            if isinstance(mask_image, (str, Path)):
                from PIL import Image
                with Image.open(mask_image) as img:
                    mask_array = np.array(img)
            else:
                mask_array = mask_image

            # Convert to binary
            if mask_array.ndim == 3:
                mask_array = np.mean(mask_array, axis=2)

            if mask_array.dtype == np.uint8:
                mask_array = mask_array.astype(np.float64) / 255.0
            elif mask_array.dtype == np.uint16:
                mask_array = mask_array.astype(np.float64) / 65535.0

            mask = mask_array > threshold

            return self.set_roi_from_mask(mask)

        except Exception as e:
            print(f"Error setting ROI from image: {e}")
            return Status.FAILED

    # Parameter management

    def set_parameters(self, params: DICParameters) -> Status:
        """
        Set DIC parameters.

        Args:
            params: DIC parameters

        Returns:
            Status
        """
        try:
            params.validate()
            self._params = params

            # Clear results if parameters changed
            self._results = None

            return Status.SUCCESS

        except Exception as e:
            print(f"Invalid parameters: {e}")
            return Status.FAILED

    # Analysis

    def calculate_seeds(self, search_radius: int = 50) -> Status:
        """
        Calculate initial seed positions.

        Args:
            search_radius: NCC search radius

        Returns:
            Status
        """
        if self._ref_img is None or not self._cur_imgs:
            print("Reference and current images must be set")
            return Status.FAILED

        if self._roi_ref is None:
            print("ROI must be set")
            return Status.FAILED

        try:
            self._report_progress(0, "Calculating seeds...")

            calculator = SeedCalculator(self._params)
            calculator.set_progress_callback(self._progress_callback)

            self._seeds = calculator.calculate_seeds(
                self._ref_img,
                self._cur_imgs[0],
                self._roi_ref,
                search_radius
            )

            self._report_progress(1.0, "Seeds calculated")
            return Status.SUCCESS

        except Exception as e:
            print(f"Error calculating seeds: {e}")
            return Status.FAILED

    def run_analysis(self) -> AnalysisResults:
        """
        Run full DIC analysis.

        Returns:
            AnalysisResults with displacements and strains
        """
        if self._ref_img is None or not self._cur_imgs:
            raise RuntimeError("Reference and current images must be set")

        if self._roi_ref is None:
            raise RuntimeError("ROI must be set")

        # Calculate seeds if not done
        if not self._seeds:
            status = self.calculate_seeds()
            if status != Status.SUCCESS:
                raise RuntimeError("Failed to calculate seeds")

        self._report_progress(0, "Running DIC analysis...")

        # Run DIC
        dic = DICAnalysis(self._params)
        dic.set_progress_callback(lambda p, m: self._report_progress(p * 0.7, m))

        displacements = dic.analyze(
            self._ref_img,
            self._cur_imgs,
            self._roi_ref,
            self._seeds
        )

        # Calculate strains
        self._report_progress(0.7, "Calculating strains...")

        strain_calc = StrainCalculator(strain_radius=5)
        strains_ref = []
        strains_cur = []

        for i, disp in enumerate(displacements):
            self._report_progress(0.7 + 0.3 * i / len(displacements), f"Strain {i+1}/{len(displacements)}")

            # Create temporary ROI from displacement result
            temp_roi = NcorrROI()
            temp_roi.set_roi("load", {"mask": disp.roi, "cutoff": 0})

            strain_ref = strain_calc.calculate_green_lagrange(
                disp.u, disp.v, temp_roi, self._params.spacing + 1
            )
            strain_cur = strain_calc.calculate_eulerian_almansi(
                disp.u, disp.v, temp_roi, self._params.spacing + 1
            )

            strains_ref.append(strain_ref)
            strains_cur.append(strain_cur)

        self._results = AnalysisResults(
            displacements=displacements,
            strains_ref=strains_ref,
            strains_cur=strains_cur,
            parameters=self._params,
        )

        self._report_progress(1.0, "Analysis complete")

        return self._results

    # Getters

    @property
    def reference_image(self) -> Optional[NcorrImage]:
        """Get reference image."""
        return self._ref_img

    @property
    def current_images(self) -> List[NcorrImage]:
        """Get current images."""
        return self._cur_imgs

    @property
    def roi(self) -> Optional[NcorrROI]:
        """Get reference ROI."""
        return self._roi_ref

    @property
    def parameters(self) -> DICParameters:
        """Get DIC parameters."""
        return self._params

    @property
    def seeds(self) -> List[SeedInfo]:
        """Get seed info."""
        return self._seeds

    @property
    def results(self) -> Optional[AnalysisResults]:
        """Get analysis results."""
        return self._results


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ncorr 2D - Digital Image Correlation"
    )
    parser.add_argument("--version", action="version", version="Ncorr 2.0.0")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI (if available)"
    )

    args = parser.parse_args()

    if args.gui:
        try:
            from .gui import launch_gui
            launch_gui()
        except ImportError:
            print("GUI not available. Install PyQt6 for GUI support.")
            print("  pip install ncorr[gui]")
    else:
        print("Ncorr 2.0.0 - Digital Image Correlation")
        print("Use --gui to launch graphical interface")
        print("Or import ncorr in Python for programmatic access")


if __name__ == "__main__":
    main()
