# Ncorr Python Scripts - Documentation

Diese Dokumentation beschreibt alle verfügbaren Python-Skripte für die DIC-Analyse (Digital Image Correlation) zur Rissausbreitung.

## Übersicht

| Skript | Beschreibung |
|--------|--------------|
| `batch_crack_analysis.py` | Haupt-Analyseskript für Bildserien |
| `plot_crack_results.py` | Visualisierung der Analyseergebnisse |
| `inspect_result.py` | Inspektion einzelner Ergebnisdateien |
| `export_statistics.py` | Export von Statistiken als ASCII-Datei |
| `register_images.py` | Bildregistrierung für perspektivische Korrektur |

---

## 1. batch_crack_analysis.py

Führt die DIC-Analyse für eine Serie von Bildern durch und berechnet Verschiebungen, Dehnungen und relative Verschiebungen.

### Verwendung

```bash
python batch_crack_analysis.py <input_dir> [optionen]
```

### Argumente

| Argument | Kurz | Typ | Standard | Beschreibung |
|----------|------|-----|----------|--------------|
| `input_dir` | | Pfad | *erforderlich* | Verzeichnis mit den zu analysierenden Bildern |
| `--output` | `-o` | Pfad | `input_dir/results` | Ausgabeverzeichnis für Ergebnisse |
| `--resolution` | | float | 20.0 | Bildauflösung in µm/pixel |
| `--reference-distance` | | float | 1.0 | Referenzabstand in mm für relative Verschiebung |
| `--subset-radius` | | int | 20 | DIC-Subset-Radius in Pixeln |
| `--subset-spacing` | | int | 3 | DIC-Subset-Abstand in Pixeln |
| `--strain-radius` | | int | 5 | Radius für Dehnungsberechnung |
| `--roi-margin` | | int | 30 | Randabstand für ROI in Pixeln |
| `--rotation` | | float | 0.0 | Post-DIC Rotation in Grad (gegen Uhrzeigersinn positiv) |
| `--no-skip` | | Flag | | Bereits verarbeitete Bilder nicht überspringen |

### Beispiel

```bash
python batch_crack_analysis.py ./images \
    --output ./results \
    --resolution 15.5 \
    --reference-distance 2.0 \
    --subset-radius 25 \
    --rotation 1.5
```

### Ausgabe

- `results/*.npz` - Ergebnisdateien für jedes Bild
- `results/config.json` - Konfigurationsparameter

---

## 2. plot_crack_results.py

Erstellt Visualisierungen aus den Analyseergebnissen: Heatmap-Videos und Schadensevolutions-Diagramme.

### Verwendung

```bash
python plot_crack_results.py <results_dir> [optionen]
```

### Argumente

| Argument | Kurz | Typ | Standard | Beschreibung |
|----------|------|-----|----------|--------------|
| `results_dir` | | Pfad | *erforderlich* | Verzeichnis mit Analyseergebnissen |
| `--output` | `-o` | Pfad | `results_dir` | Ausgabeverzeichnis für Plots |
| `--video-fps` | | int | 10 | Frames pro Sekunde für Video |
| `--dpi` | | int | 150 | Auflösung für Plots |
| `--no-video` | | Flag | | Video-Generierung überspringen |
| `--video-field` | | str | `relative_v` | Feld für Video-Darstellung |
| `--pixels` | | Flag | | Pixel-Koordinaten statt mm verwenden |
| `--crack-margin` | | float | 2.0 | Rand um Rissbereich in mm |
| `--sigma-x` | | float | 3.0 | Gauss-Glättung Sigma entlang x-Achse |
| `--sigma-time` | | float | 2.0 | Gauss-Glättung Sigma entlang Zeitachse |
| `--smoothing` | | str | `gaussian` | Glättungsmethode: `gaussian` oder `isotonic` |
| `--title` | | str | | Benutzerdefinierter Titel für Plots |

### Video-Feld Optionen

- `displacement_u` - Verschiebung in x-Richtung
- `displacement_v` - Verschiebung in y-Richtung
- `strain_exx` - Dehnung εxx
- `strain_eyy` - Dehnung εyy
- `strain_exy` - Dehnung εxy
- `relative_u` - Relative Verschiebung in x
- `relative_v` - Relative Verschiebung in y

### Beispiel

```bash
python plot_crack_results.py ./results \
    --video-field strain_eyy \
    --smoothing isotonic \
    --video-fps 15 \
    --dpi 200
```

### Ausgabe

- `heatmap_<field>.mp4` - Heatmap-Video
- `damage_evolution.png` - Schadensevolutions-Diagramm
- `damage_ratio.npy` - Schadensverhältnis als NumPy-Array

---

## 3. inspect_result.py

Zeigt alle Felder einer einzelnen Ergebnisdatei zur Diagnose und Qualitätskontrolle.

### Verwendung

```bash
python inspect_result.py <result_file> [optionen]
```

### Argumente

| Argument | Kurz | Typ | Standard | Beschreibung |
|----------|------|-----|----------|--------------|
| `result_file` | | Pfad | *erforderlich* | Pfad zur .npz Ergebnisdatei |
| `--save` | `-s` | Pfad | | Figur in Datei speichern statt anzeigen |
| `--dpi` | | int | 150 | DPI für gespeicherte Figur |
| `--image-dir` | `-i` | Pfad | | Verzeichnis mit Originalbildern |
| `--raw` | | Flag | | Rohe .npz Dateiinhalte ausgeben |

### Beispiel

```bash
python inspect_result.py ./results/image_005.npz \
    --image-dir ./images \
    --save inspection.png
```

### Darstellung

- **Verschiebungen**: viridis Farbskala
- **Dehnungen**: RdBu_r Farbskala (rot-weiß-blau), Werte in %
- **Seitenverhältnis**: wird beibehalten

### Layout

```
Row 1: displacement_u | displacement_v | original_image
Row 2: strain_exx     | strain_eyy     | strain_exy
Row 3: relative_u     | relative_v     | (leer)
Row 4: max_rel_v(x)   | y_position     | valid_mask
```

---

## 4. export_statistics.py

Exportiert Statistiken (Min, Max, Mittelwert, Standardabweichung) aller Ergebnisse als Tab-separierte ASCII-Datei.

### Verwendung

```bash
python export_statistics.py <results_dir> [optionen]
```

### Argumente

| Argument | Kurz | Typ | Standard | Beschreibung |
|----------|------|-----|----------|--------------|
| `results_dir` | | Pfad | *erforderlich* | Verzeichnis mit .npz Ergebnisdateien |
| `--output` | `-o` | Pfad | `results_dir/statistics.txt` | Ausgabedatei |
| `--summary` | | Flag | | Globale Zusammenfassung ausgeben |

### Beispiel

```bash
python export_statistics.py ./results \
    --output ./stats.txt \
    --summary
```

### Ausgabeformat

Tab-separierte Datei mit folgenden Spalten pro Bild:

| Spalte | Einheit | Beschreibung |
|--------|---------|--------------|
| `index` | - | Bildnummer |
| `image_name` | - | Dateiname |
| `displacement_u_min/max/mean/std` | px | Verschiebung u |
| `displacement_v_min/max/mean/std` | px | Verschiebung v |
| `strain_exx_min/max/mean/std` | % | Dehnung εxx |
| `strain_eyy_min/max/mean/std` | % | Dehnung εyy |
| `strain_exy_min/max/mean/std` | % | Dehnung εxy |
| `relative_u_min/max/mean/std` | px | Relative Verschiebung u |
| `relative_v_min/max/mean/std` | px | Relative Verschiebung v |
| `max_relative_v_min/max/mean/std` | px | Max. relative Verschiebung v |

---

## 5. register_images.py

Registriert zwei Bilder unter Verwendung korrespondierender Punkte. Korrigiert Verschiebung, Rotation und Skalierung.

### Verwendung

```bash
python register_images.py <image1> <image2> [optionen]
```

### Argumente

| Argument | Kurz | Typ | Standard | Beschreibung |
|----------|------|-----|----------|--------------|
| `image1` | | Pfad | *erforderlich* | Referenzbild (Zielgeometrie) |
| `image2` | | Pfad | *erforderlich* | Zu transformierendes Bild |
| `--output` | `-o` | Pfad | `image2_registered.ext` | Ausgabepfad für registriertes Bild |
| `--interactive` | `-i` | Flag | | Interaktive Punktauswahl |
| `--points` | `-p` | str[] | | Punktpaare im Format `x1,y1:x2,y2` |
| `--regions` | `-r` | str[] | | Regionen für Template-Matching im Format `x,y,radius` |
| `--search-margin` | | int | 100 | Suchbereich für Template-Matching in Pixeln |
| `--min-quality` | | float | 0.5 | Minimale Match-Qualität (0.0-1.0) |
| `--num-points` | `-n` | int | 3 | Anzahl Punkte im interaktiven Modus |
| `--visualize` | `-v` | Flag | | Visualisierung anzeigen |
| `--save-visualization` | | Pfad | | Visualisierung in Datei speichern |

### Koordinatensystem

Pixel werden von **oben links** gezählt:
- **x**: horizontal, wächst nach rechts
- **y**: vertikal, wächst nach unten

```
(0,0) ----→ x
  |
  ↓
  y
```

### Methoden

#### 1. Interaktiver Modus

```bash
python register_images.py ref.png target.png --interactive --visualize
```

Klicken Sie nacheinander auf 3 korrespondierende Punkte in beiden Bildern.

#### 2. Template-Matching

```bash
python register_images.py ref.png target.png \
    --regions "100,100,25" "500,100,25" "300,400,25" \
    --search-margin 150 \
    --min-quality 0.3 \
    --visualize
```

Format: `x,y,radius` - Mittelpunkt und Radius der Region.

#### 3. Vordefinierte Punkte

```bash
python register_images.py ref.png target.png \
    --points "100,100:102,98" "500,100:503,97" "300,400:305,402"
```

Format: `x1,y1:x2,y2` - Punkt in Bild1 : entsprechender Punkt in Bild2.

### Ausgabe

- `*_registered.<ext>` - Transformiertes Bild
- `*_transform.txt` - Transformationsmatrix

### Berechnete Parameter

Bei 3 Punkten (affine Transformation):
- Skalierung X und Y
- Rotation in Grad
- Translation in Pixeln

---

## Typischer Workflow

```bash
# 1. Optional: Bilder registrieren (falls Kameraposition geändert)
python register_images.py ref.png target.png --interactive -v

# 2. DIC-Analyse durchführen
python batch_crack_analysis.py ./images \
    --resolution 15.5 \
    --reference-distance 2.0

# 3. Einzelnes Ergebnis inspizieren
python inspect_result.py ./images/results/image_010.npz --image-dir ./images

# 4. Statistiken exportieren
python export_statistics.py ./images/results --summary

# 5. Visualisierungen erstellen
python plot_crack_results.py ./images/results --video-field strain_eyy
```

---

## Abhängigkeiten

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- OpenCV (cv2)

Installation:
```bash
pip install numpy scipy matplotlib opencv-python
```
