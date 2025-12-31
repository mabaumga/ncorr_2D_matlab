# Grundkonzepte

## Digital Image Correlation (DIC)

### Prinzip

DIC basiert auf dem Vergleich von Grauwertmustern zwischen einem Referenzbild und einem verformten Bild. Das Verfahren:

1. Teilt das Referenzbild in kleine **Subsets** (Teilbereiche) auf
2. Sucht jedes Subset im verformten Bild
3. Berechnet die lokale Verschiebung und Dehnungsgradienten

### Subsets und Korrelation

```
Referenzbild                 Verformtes Bild
┌─────────────┐              ┌─────────────┐
│             │              │             │
│   ┌───┐     │              │      ┌───┐  │
│   │ S │     │   ──────►    │      │ S'│  │
│   └───┘     │              │      └───┘  │
│   (x,y)     │              │   (x+u, y+v)│
└─────────────┘              └─────────────┘
```

**S** = Subset im Referenzbild bei Position (x, y)
**S'** = Verschobenes Subset bei Position (x+u, y+v)

### Korrelationskoeffizient

Der **Zero-Mean Normalized Cross-Correlation (ZNCC)** Koeffizient:

```
         Σ[(f - f̄)(g - ḡ)]
C = ─────────────────────────────
    √[Σ(f - f̄)²] √[Σ(g - ḡ)²]
```

- **C = 1**: Perfekte Übereinstimmung
- **C = 0**: Keine Korrelation
- **C < 0**: Inverse Korrelation

## Parameter verstehen

### Subset-Radius (radius)

Der Radius definiert die Größe des Korrelationsfensters.

```
       ◄─────── 2×radius + 1 ───────►
      ┌─────────────────────────────┐
      │                             │
      │                             │
      │            (x,y)            │ ▲
      │              ●              │ │
      │                             │ 2×radius + 1
      │                             │ │
      │                             │ ▼
      └─────────────────────────────┘
```

**Typische Werte:** 15-50 Pixel

| Radius | Vorteile | Nachteile |
|--------|----------|-----------|
| Klein (15-25) | Hohe räumliche Auflösung | Empfindlich gegenüber Rauschen |
| Mittel (25-40) | Guter Kompromiss | - |
| Groß (40-60) | Robust, gute Korrelation | Niedrige räumliche Auflösung |

**Faustregel:** Der Radius sollte 3-5 Speckle-Durchmesser umfassen.

### Gitterabstand (spacing)

Der Abstand zwischen den Analysepunkten.

```
   spacing = 3

   ●  ·  ·  ●  ·  ·  ●  ·  ·  ●
   ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
   ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
   ●  ·  ·  ●  ·  ·  ●  ·  ·  ●
   ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
   ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
   ●  ·  ·  ●  ·  ·  ●  ·  ·  ●

   ● = Analysepunkt
   · = Übersprungen
```

**Typische Werte:** 1-10 Pixel

- **spacing=0**: Jedes Pixel wird analysiert (langsam, höchste Auflösung)
- **spacing=5**: Jedes 6. Pixel wird analysiert (schneller)

### Konvergenzparameter

#### cutoff_diffnorm

Die Konvergenzschwelle für die iterative Optimierung.

```python
cutoff_diffnorm = 1e-3  # Standard
```

- Kleinere Werte → Höhere Genauigkeit, aber längere Rechenzeit
- Typischer Bereich: 1e-4 bis 1e-2

#### cutoff_iteration

Maximale Anzahl an Optimierungsiterationen.

```python
cutoff_iteration = 30  # Standard
```

- 20-50 Iterationen sind typisch
- Bei Konvergenzproblemen erhöhen

## Region of Interest (ROI)

### Was ist eine ROI?

Die ROI definiert den Bereich des Bildes, der analysiert wird:

```
┌───────────────────────────────────┐
│                                   │
│       ┌─────────────────┐         │
│       │                 │         │
│       │    ROI (weiß)   │         │
│       │                 │         │
│       └─────────────────┘         │
│                                   │
│   Hintergrund (schwarz)           │
└───────────────────────────────────┘
```

### ROI als Maske

Eine ROI wird als binäres NumPy-Array definiert:

```python
import numpy as np

# Rechteckige ROI
mask = np.zeros((height, width), dtype=bool)
mask[100:500, 100:700] = True

# Kreisförmige ROI
y, x = np.ogrid[:height, :width]
center = (height//2, width//2)
radius = 200
mask = ((x - center[1])**2 + (y - center[0])**2) <= radius**2

# ROI aus Bild laden
from PIL import Image
mask_img = np.array(Image.open("maske.png"))
mask = mask_img > 127  # Schwellwert
```

### Regionen

Eine ROI kann mehrere getrennte Regionen enthalten:

```python
from ncorr.core.roi import NcorrROI

roi = NcorrROI()
roi.set_roi("load", {"mask": mask, "cutoff": 20})

print(f"Anzahl Regionen: {len(roi.regions)}")
for i, region in enumerate(roi.regions):
    print(f"Region {i}: {region.totalpoints} Punkte")
```

## Dehnungsberechnung

### Verschiebungsgradient

Die Dehnung wird aus den Verschiebungsgradienten berechnet:

```
∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y
```

Diese werden mit zentralen Differenzen approximiert:

```
∂u/∂x ≈ [u(x+r) - u(x-r)] / (2r)
```

Wobei **r** der Dehnungsradius ist.

### Green-Lagrange-Dehnung

Die Green-Lagrange-Dehnung bezieht sich auf die **Referenzkonfiguration**:

```
Exx = ∂u/∂x + 0.5 × [(∂u/∂x)² + (∂v/∂x)²]
Eyy = ∂v/∂y + 0.5 × [(∂u/∂y)² + (∂v/∂y)²]
Exy = 0.5 × (∂u/∂y + ∂v/∂x + ∂u/∂x×∂u/∂y + ∂v/∂x×∂v/∂y)
```

**Verwendung:** Typisch für Finite-Elemente-Vergleiche

### Eulerian-Almansi-Dehnung

Die Eulerian-Almansi-Dehnung bezieht sich auf die **aktuelle Konfiguration**:

```
exx = ∂u/∂x - 0.5 × [(∂u/∂x)² + (∂v/∂x)²]
eyy = ∂v/∂y - 0.5 × [(∂u/∂y)² + (∂v/∂y)²]
exy = 0.5 × (∂u/∂y + ∂v/∂x - ∂u/∂x×∂u/∂y - ∂v/∂x×∂v/∂y)
```

### Dehnungsradius

Der Dehnungsradius beeinflusst die Glättung der Dehnungsfelder:

```python
from ncorr.algorithms.strain import StrainCalculator

# Kleiner Radius (detaillierter, mehr Rauschen)
calc_detail = StrainCalculator(strain_radius=3)

# Großer Radius (glatter, weniger Details)
calc_smooth = StrainCalculator(strain_radius=10)
```

## B-Spline-Interpolation

### Warum B-Splines?

Ncorr verwendet biquintische B-Splines für:

1. **Subpixel-Genauigkeit**: Interpolation zwischen Pixeln
2. **Glatte Gradienten**: Analytische Ableitungen möglich
3. **Hohe Genauigkeit**: Besser als bilineare Interpolation

### Wie funktioniert es?

```
Originalbild → B-Spline-Koeffizienten → Interpolierter Wert

   [Pixel]          [bcoef]           [Grauwert bei (x.5, y.3)]
```

```python
# Automatisch bei Bildladung
img = NcorrImage.from_file("bild.tif")
bcoef = img.get_bcoef()  # B-Spline-Koeffizienten
```

## Seed-Punkte

### Was sind Seeds?

Seeds sind Startpunkte für die DIC-Analyse. Von jedem Seed aus breitet sich die Analyse per Flood-Fill aus.

```
       Region 1              Region 2
    ┌─────────────┐      ┌─────────────┐
    │             │      │             │
    │      ×      │      │      ×      │
    │    Seed 1   │      │    Seed 2   │
    │             │      │             │
    └─────────────┘      └─────────────┘
```

### Automatische Seed-Berechnung

```python
# Seeds werden automatisch berechnet
ncorr.calculate_seeds(search_radius=50)

# Oder manuell setzen
from ncorr.algorithms.dic import SeedInfo

seeds = [
    SeedInfo(x=100, y=100, u=0, v=0, region_idx=0, valid=True),
    SeedInfo(x=300, y=200, u=0, v=0, region_idx=1, valid=True),
]
```

### Seed-Suche

Die Seed-Suche verwendet:
1. **NCC-Suche**: Grobe Lokalisierung mit Template-Matching
2. **IC-GN-Verfeinerung**: Subpixel-genaue Positionierung

## Step-Analyse für große Verformungen

Bei großen Verformungen kann die Korrelation versagen. Die Step-Analyse hilft:

```python
from ncorr.core.dic_parameters import DICParameters, StepAnalysis

params = DICParameters(
    radius=30,
    spacing=5,
)
params.step_analysis = StepAnalysis(
    enabled=True,
    type="regular",  # oder "backward"
    auto=True,
    step=1,
)
```

### Funktionsweise

```
Bild 0 (Ref)  →  Bild 1  →  Bild 2  →  Bild 3
     │              │           │           │
     └──────────────┘           │           │
          Seed von               │           │
          Bild 0→1               │           │
                    └───────────┘           │
                         Seed von            │
                         Bild 1→2            │
                                  └──────────┘
                                       Seed von
                                       Bild 2→3
```

Die Verschiebung aus dem vorherigen Bildpaar wird als Startpunkt für das nächste verwendet.
