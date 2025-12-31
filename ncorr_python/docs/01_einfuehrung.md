# Einführung in Ncorr 2D Python

## Überblick

Ncorr 2D ist eine Software zur Durchführung von Digital Image Correlation (DIC) Analysen. Diese Python-Version bietet die gleiche Funktionalität wie die originale MATLAB-Version, jedoch mit den Vorteilen von Python:

- Kostenlos und Open Source
- Einfache Integration in bestehende Python-Workflows
- Zugang zu umfangreichen Python-Bibliotheken
- Moderne Programmierschnittstelle

## Hauptfunktionen

### 1. Verschiebungsmessung

Ncorr berechnet Verschiebungsfelder (u, v) zwischen einem Referenzbild und einem oder mehreren verformten Bildern mit Subpixel-Genauigkeit.

### 2. Dehnungsberechnung

Aus den Verschiebungsfeldern werden Dehnungen berechnet:
- **Green-Lagrange-Dehnung** (Referenzkonfiguration)
- **Eulerian-Almansi-Dehnung** (aktuelle Konfiguration)

### 3. Region of Interest (ROI)

Definieren Sie präzise Analysebereiche über:
- Binäre Masken
- Bildbasierte Schwellwerte
- Geometrische Formen

## Algorithmus

Ncorr verwendet den **Inverse Compositional Gauss-Newton (IC-GN)** Algorithmus für die Korrelation:

1. **Referenz-Template**: Ein kreisförmiges Subset um jeden Analysepunkt
2. **Suchverfahren**: Finden der besten Übereinstimmung im verformten Bild
3. **Subpixel-Verfeinerung**: Iterative Optimierung für höchste Genauigkeit

### Mathematische Grundlagen

Die Verschiebung wird durch Minimierung des Korrelationskoeffizienten bestimmt:

```
C = 1 - Σ[(f - f̄)(g - ḡ)] / √[Σ(f - f̄)² Σ(g - ḡ)²]
```

Wobei:
- f = Grauwerte im Referenzbild
- g = Grauwerte im verformten Bild
- f̄, ḡ = Mittelwerte

## Systemanforderungen

### Software
- Python 3.9 oder höher
- NumPy, SciPy, Pillow
- Numba (für Beschleunigung)

### Hardware
- RAM: Mindestens 4 GB (8 GB empfohlen für große Bilder)
- CPU: Mehrkernprozessor empfohlen

## Projektstruktur

```
ncorr/
├── core/               # Kernklassen
│   ├── image.py       # Bildverarbeitung
│   ├── roi.py         # Region of Interest
│   └── dic_parameters.py  # Parameter
├── algorithms/         # Algorithmen
│   ├── dic.py         # DIC-Analyse
│   ├── strain.py      # Dehnungsberechnung
│   └── bspline.py     # B-Spline Interpolation
├── utils/             # Hilfsfunktionen
└── main.py            # Haupt-API
```

## Nächste Schritte

- [Installation](02_installation.md) - Software installieren
- [Schnellstart](03_schnellstart.md) - Erste Analyse durchführen
- [Grundkonzepte](04_grundkonzepte.md) - Tieferes Verständnis
