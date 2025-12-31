# Ncorr 2D Python - Dokumentation

Willkommen zur Dokumentation der Python-Version von Ncorr 2D, einer Open-Source-Software für Digital Image Correlation (DIC).

## Inhaltsverzeichnis

1. [Einführung](01_einfuehrung.md)
2. [Installation](02_installation.md)
3. [Schnellstart](03_schnellstart.md)
4. [Grundkonzepte](04_grundkonzepte.md)
5. [Tutorials](05_tutorials.md)
6. [API-Referenz](06_api_referenz.md)
7. [Beispiele](07_beispiele.md)
8. [Troubleshooting](08_troubleshooting.md)

## Was ist Digital Image Correlation (DIC)?

Digital Image Correlation ist eine optische Messmethode zur Bestimmung von Verschiebungs- und Dehnungsfeldern auf Oberflächen. Dabei werden Bilder einer Probe vor und nach der Verformung aufgenommen und miteinander korreliert, um die lokalen Verschiebungen zu berechnen.

### Anwendungsgebiete

- Materialprüfung und -charakterisierung
- Bruchmechanik
- Biomechanik
- Geotechnik
- Luft- und Raumfahrt
- Automobilindustrie

## Schnellübersicht

```python
from ncorr import Ncorr, DICParameters

# Ncorr initialisieren
ncorr = Ncorr()

# Bilder laden
ncorr.set_reference("referenz.tif")
ncorr.set_current(["verformt_001.tif", "verformt_002.tif"])

# Region of Interest definieren
ncorr.set_roi_from_mask(mask_array)

# Parameter konfigurieren
ncorr.set_parameters(DICParameters(radius=30, spacing=5))

# Analyse durchführen
results = ncorr.run_analysis()

# Ergebnisse verwenden
u = results.displacements[0].u  # u-Verschiebung
v = results.displacements[0].v  # v-Verschiebung
```

## Lizenz

Diese Software ist eine Übersetzung der originalen MATLAB-Version von Ncorr.

**Referenz:**
> Ncorr: open-source 2D digital image correlation matlab software
> J Blaber, B Adair, A Antoniou
> Experimental Mechanics 55 (6), 1105-1122
