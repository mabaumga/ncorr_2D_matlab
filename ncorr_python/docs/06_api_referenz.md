# API-Referenz

Vollständige Referenz aller Klassen und Funktionen in Ncorr Python.

## Hauptklasse

### `Ncorr`

Die Hauptklasse für DIC-Analysen.

```python
from ncorr import Ncorr

ncorr = Ncorr()
```

#### Methoden

##### `set_reference(source, lazy=False) → Status`

Setzt das Referenzbild.

**Parameter:**
- `source`: Pfad (str/Path), NumPy-Array oder NcorrImage
- `lazy`: Wenn True, wird das Bild erst bei Bedarf geladen

**Rückgabe:** `Status.SUCCESS` oder `Status.FAILED`

```python
# Aus Datei
ncorr.set_reference("referenz.tif")

# Aus Array
ncorr.set_reference(numpy_array)

# Lazy Loading (speichereffizient)
ncorr.set_reference("großes_bild.tif", lazy=True)
```

##### `set_current(sources, lazy=False) → Status`

Setzt das/die verformte(n) Bild(er).

**Parameter:**
- `sources`: Einzelnes Bild oder Liste von Bildern
- `lazy`: Lazy Loading aktivieren

```python
# Einzelnes Bild
ncorr.set_current("verformt.tif")

# Mehrere Bilder
ncorr.set_current(["bild_001.tif", "bild_002.tif", "bild_003.tif"])
```

##### `set_roi_from_mask(mask, min_region_size=20) → Status`

Setzt die ROI aus einer binären Maske.

**Parameter:**
- `mask`: NumPy boolean Array
- `min_region_size`: Minimale Pixelanzahl für gültige Regionen

```python
mask = np.ones((height, width), dtype=bool)
mask[:50, :] = False  # Rand ausschließen
ncorr.set_roi_from_mask(mask)
```

##### `set_roi_from_image(mask_image, threshold=0.5) → Status`

Lädt ROI aus einem Maskenbild.

**Parameter:**
- `mask_image`: Pfad zum Maskenbild
- `threshold`: Schwellwert für Binarisierung

```python
ncorr.set_roi_from_image("maske.png", threshold=0.5)
```

##### `set_parameters(params) → Status`

Setzt die DIC-Parameter.

**Parameter:**
- `params`: DICParameters-Objekt

```python
from ncorr import DICParameters

params = DICParameters(radius=30, spacing=5)
ncorr.set_parameters(params)
```

##### `calculate_seeds(search_radius=50) → Status`

Berechnet Seed-Punkte für jede Region.

**Parameter:**
- `search_radius`: Suchradius für NCC

```python
ncorr.calculate_seeds(search_radius=50)
```

##### `run_analysis() → AnalysisResults`

Führt die vollständige DIC-Analyse durch.

**Rückgabe:** `AnalysisResults` mit Verschiebungen und Dehnungen

```python
results = ncorr.run_analysis()
```

##### `set_progress_callback(callback)`

Setzt Callback für Fortschrittsanzeige.

**Parameter:**
- `callback`: Funktion mit Signatur `(progress: float, message: str) → None`

```python
def show_progress(progress, message):
    print(f"{progress*100:.0f}% - {message}")

ncorr.set_progress_callback(show_progress)
```

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|-------------|-----|--------------|
| `reference_image` | `NcorrImage` | Referenzbild |
| `current_images` | `List[NcorrImage]` | Verformte Bilder |
| `roi` | `NcorrROI` | Region of Interest |
| `parameters` | `DICParameters` | DIC-Parameter |
| `seeds` | `List[SeedInfo]` | Seed-Punkte |
| `results` | `AnalysisResults` | Analyseergebnisse |

---

## Core-Klassen

### `NcorrImage`

Repräsentiert ein Bild mit B-Spline-Koeffizienten.

```python
from ncorr.core.image import NcorrImage

# Aus Datei laden
img = NcorrImage.from_file("bild.tif")

# Aus Array erstellen
img = NcorrImage.from_array(numpy_array, name="mein_bild")
```

#### Methoden

| Methode | Rückgabe | Beschreibung |
|---------|----------|--------------|
| `get_img()` | `ndarray[uint8]` | RGB-Bild (H×W×3) |
| `get_gs()` | `ndarray[float64]` | Grauwerte (H×W), Bereich [0,1] |
| `get_bcoef()` | `ndarray[float64]` | B-Spline-Koeffizienten |
| `reduce(spacing)` | `NcorrImage` | Reduziertes Bild |
| `formatted()` | `dict` | Als Dictionary |

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|-------------|-----|--------------|
| `is_set` | `bool` | Wurde gesetzt? |
| `height` | `int` | Höhe in Pixel |
| `width` | `int` | Breite in Pixel |
| `name` | `str` | Dateiname |
| `path` | `str` | Dateipfad |
| `border_bcoef` | `int` | Randbreite (default: 20) |

---

### `NcorrROI`

Region of Interest mit Regionen und Grenzen.

```python
from ncorr.core.roi import NcorrROI

roi = NcorrROI()
roi.set_roi("load", {"mask": mask, "cutoff": 20})
```

#### Methoden

##### `set_roi(roi_type, data)`

Setzt die ROI.

**Parameter:**
- `roi_type`: "load", "draw", "region", oder "boundary"
- `data`: Dictionary mit typ-spezifischen Daten

```python
# Aus Maske
roi.set_roi("load", {"mask": mask, "cutoff": 20})

# Aus Regionen
roi.set_roi("region", {"region": regions, "size_mask": (height, width)})
```

##### `reduce(spacing) → NcorrROI`

Reduziert die ROI.

##### `get_num_region(x, y, skip_regions=None) → (int, int)`

Findet die Region, die Punkt (x, y) enthält.

**Rückgabe:** (region_index, nodelist_index) oder (-1, 0)

##### `get_region_mask(num_region) → ndarray[bool]`

Gibt Maske für eine einzelne Region zurück.

##### `get_circular_roi(x, y, radius, subset_trunc=False) → CircularROI`

Erstellt kreisförmiges Subset.

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|-------------|-----|--------------|
| `is_set` | `bool` | Wurde gesetzt? |
| `roi_type` | `ROIType` | Art der ROI |
| `mask` | `ndarray[bool]` | Binäre Maske |
| `regions` | `List[Region]` | Verbundene Regionen |
| `boundaries` | `List[Boundary]` | Grenzkonturen |

---

### `Region`

Einzelne verbundene Region in einer ROI.

```python
from ncorr.core.roi import Region

region = Region(
    nodelist=np.array([[10, 50], [15, 55]], dtype=np.int32),
    noderange=np.array([2, 2], dtype=np.int32),
    leftbound=0,
    rightbound=1,
    upperbound=10,
    lowerbound=55,
    totalpoints=82,
)
```

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|-------------|-----|--------------|
| `nodelist` | `ndarray[int32]` | Y-Koordinatenpaare pro Spalte |
| `noderange` | `ndarray[int32]` | Anzahl Einträge pro Spalte |
| `leftbound` | `int` | Linke X-Grenze |
| `rightbound` | `int` | Rechte X-Grenze |
| `upperbound` | `int` | Obere Y-Grenze |
| `lowerbound` | `int` | Untere Y-Grenze |
| `totalpoints` | `int` | Gesamtzahl Pixel |

---

### `DICParameters`

Konfiguration für DIC-Analyse.

```python
from ncorr import DICParameters

params = DICParameters(
    radius=30,
    spacing=5,
    cutoff_diffnorm=1e-3,
    cutoff_iteration=30,
    total_threads=1,
    subset_trunc=False,
    pix_to_units=1.0,
    units="pixels",
)
```

#### Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `radius` | `int` | 30 | Subset-Radius (10-200) |
| `spacing` | `int` | 5 | Gitterabstand (0-80) |
| `cutoff_diffnorm` | `float` | 1e-2 | Konvergenzschwelle (1e-8 bis 1) |
| `cutoff_iteration` | `int` | 20 | Max. Iterationen (5-100) |
| `total_threads` | `int` | 1 | Anzahl Threads |
| `subset_trunc` | `bool` | False | Subset-Trunkierung |
| `pix_to_units` | `float` | 1.0 | Pixel-zu-Einheit-Faktor |
| `units` | `str` | "pixels" | Einheitenbezeichnung |
| `lens_coef` | `float` | 0.0 | Linsenverzerrungskoeffizient |
| `step_analysis` | `StepAnalysis` | - | Step-Analyse-Einstellungen |

#### Methoden

| Methode | Beschreibung |
|---------|--------------|
| `validate()` | Prüft Parametergültigkeit |
| `to_dict()` | Konvertiert zu Dictionary |
| `from_dict(d)` | Erstellt aus Dictionary |

---

### `Status`

Enumeration für Operationsstatus.

```python
from ncorr.core.status import Status

if status == Status.SUCCESS:
    print("Erfolgreich")
elif status == Status.FAILED:
    print("Fehlgeschlagen")
elif status == Status.CANCELLED:
    print("Abgebrochen")
```

| Wert | Integer | Beschreibung |
|------|---------|--------------|
| `SUCCESS` | 1 | Erfolgreich |
| `FAILED` | 0 | Fehlgeschlagen |
| `CANCELLED` | -1 | Abgebrochen |

---

## Ergebnisklassen

### `AnalysisResults`

Vollständige Analyseergebnisse.

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|-------------|-----|--------------|
| `displacements` | `List[DICResult]` | Verschiebungen pro Bild |
| `strains_ref` | `List[StrainResult]` | Green-Lagrange-Dehnungen |
| `strains_cur` | `List[StrainResult]` | Eulerian-Almansi-Dehnungen |
| `parameters` | `DICParameters` | Verwendete Parameter |

#### Methoden

| Methode | Beschreibung |
|---------|--------------|
| `save(filepath)` | Speichert als .npz + .json |
| `load(filepath)` | Lädt gespeicherte Ergebnisse |

---

### `DICResult`

Verschiebungsergebnis für ein Bildpaar.

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|-------------|-----|--------------|
| `u` | `ndarray[float64]` | U-Verschiebung (Pixel) |
| `v` | `ndarray[float64]` | V-Verschiebung (Pixel) |
| `corrcoef` | `ndarray[float64]` | Korrelationskoeffizient |
| `roi` | `ndarray[bool]` | Gültige Punkte |
| `seed_info` | `List[SeedInfo]` | Seed-Informationen |
| `converged` | `ndarray[bool]` | Konvergenzstatus |

---

### `StrainResult`

Dehnungsergebnis.

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|-------------|-----|--------------|
| `exx` | `ndarray[float64]` | εxx-Komponente |
| `exy` | `ndarray[float64]` | εxy-Komponente (Scherung) |
| `eyy` | `ndarray[float64]` | εyy-Komponente |
| `roi` | `ndarray[bool]` | Gültige Punkte |
| `dudx` | `ndarray[float64]` | ∂u/∂x |
| `dudy` | `ndarray[float64]` | ∂u/∂y |
| `dvdx` | `ndarray[float64]` | ∂v/∂x |
| `dvdy` | `ndarray[float64]` | ∂v/∂y |

---

## Algorithmen

### `StrainCalculator`

Dehnungsberechnung aus Verschiebungsfeldern.

```python
from ncorr.algorithms.strain import StrainCalculator

calc = StrainCalculator(strain_radius=5)
strain = calc.calculate_green_lagrange(u, v, roi, spacing)
```

#### Methoden

##### `calculate_green_lagrange(u, v, roi, spacing=1) → StrainResult`

Berechnet Green-Lagrange-Dehnung (Referenzkonfiguration).

##### `calculate_eulerian_almansi(u, v, roi, spacing=1) → StrainResult`

Berechnet Eulerian-Almansi-Dehnung (aktuelle Konfiguration).

##### `calculate_principal_strains(exx, exy, eyy) → (e1, e2, theta)` [static]

Berechnet Hauptdehnungen und -richtung.

##### `calculate_von_mises(exx, exy, eyy) → ndarray` [static]

Berechnet von-Mises-Vergleichsdehnung.

---

### `BSplineInterpolator`

B-Spline-Interpolation für Subpixel-Genauigkeit.

```python
from ncorr.algorithms.bspline import BSplineInterpolator

# Koeffizienten berechnen
bcoef = BSplineInterpolator.compute_bcoef(data)

# Werte interpolieren
values = BSplineInterpolator.interpolate_values(
    points, bcoef, left_offset, top_offset, border
)
```

---

## Hilfsfunktionen

### Image Loading

```python
from ncorr.utils.image_loader import load_images, validate_image_format

# Bilder laden
images = load_images(["bild1.tif", "bild2.tif"])

# Format prüfen
valid, message = validate_image_format(numpy_array)
```

### Validierung

```python
from ncorr.utils.validation import is_real_bounded, is_int_bounded

# Reelle Zahl prüfen
valid, value, msg = is_real_bounded(0.5, 0, 1)

# Integer prüfen
valid, value, msg = is_int_bounded(30, 10, 200)
```

### Colormaps

```python
from ncorr.utils.colormaps import get_ncorr_colormap, apply_colormap

# Colormap abrufen
cmap = get_ncorr_colormap("ncorr")

# Auf Daten anwenden
rgba = apply_colormap(data, vmin=0, vmax=1, cmap_name="jet")
```
