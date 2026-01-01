# Troubleshooting

Lösungen für häufige Probleme bei der Verwendung von Ncorr Python.

## Installationsprobleme

### Problem: `ModuleNotFoundError: No module named 'ncorr'`

**Ursache:** Ncorr ist nicht im Python-Pfad.

**Lösung:**
```bash
# Installation überprüfen
pip show ncorr

# Neu installieren
cd /pfad/zu/ncorr_python
pip install -e .

# Oder Pfad manuell hinzufügen
import sys
sys.path.insert(0, '/pfad/zu/ncorr_python')
```

### Problem: Numba-Kompilierungsfehler

**Symptom:**
```
numba.core.errors.TypingError: Failed in nopython mode
```

**Lösung:**
```bash
# Numba-Cache löschen
rm -rf ~/.cache/numba

# Oder in Python
import numba
print(numba.config.CACHE_DIR)
# Dann dieses Verzeichnis löschen
```

### Problem: OpenCV-Fehler auf Server ohne Display

**Symptom:**
```
cv2.error: OpenCV cannot find a display
```

**Lösung:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

---

## Bildlade-Probleme

### Problem: `ValueError: Invalid image format`

**Ursache:** Bildformat wird nicht unterstützt.

**Lösung:**
```python
# Unterstützte Formate prüfen
from ncorr.utils.image_loader import SUPPORTED_FORMATS
print(SUPPORTED_FORMATS)
# {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Bild konvertieren
from PIL import Image
img = Image.open("bild.webp")
img.save("bild.png")
```

### Problem: CMYK-Bild (4 Kanäle)

**Symptom:**
```
ValueError: CMYK images (4 channels) are not supported
```

**Lösung:**
```python
from PIL import Image

# In RGB konvertieren
img = Image.open("cmyk_bild.tif")
rgb_img = img.convert("RGB")
rgb_img.save("rgb_bild.tif")
```

### Problem: Lazy Loading schlägt fehl

**Symptom:**
```
FileNotFoundError: Image could not be located
```

**Ursache:** Dateipfad hat sich geändert.

**Lösung:**
```python
# Vollständigen Pfad verwenden
from pathlib import Path
img_path = Path("bilder/ref.tif").resolve()
ncorr.set_reference(str(img_path))
```

---

## ROI-Probleme

### Problem: `RuntimeError: ROI has not been set yet`

**Ursache:** ROI wurde vor der Verwendung nicht gesetzt.

**Lösung:**
```python
# Prüfen ob ROI gesetzt
if ncorr.roi is None or not ncorr.roi.is_set:
    mask = np.ones((height, width), dtype=bool)
    ncorr.set_roi_from_mask(mask)
```

### Problem: Keine Regionen gefunden

**Symptom:**
```
len(roi.regions) == 0
```

**Ursache:**
- Maske ist leer
- Regionen zu klein (unter cutoff)

**Lösung:**
```python
# Maske prüfen
print(f"Maske enthält {np.sum(mask)} True-Werte")

# Cutoff reduzieren
roi.set_roi("load", {"mask": mask, "cutoff": 5})  # Statt 20
```

### Problem: ROI-Regionen werden abgeschnitten

**Ursache:** Subset-Radius ragt über ROI-Rand.

**Lösung:**
```python
# ROI-Rand erweitern
border = params.radius + 10
mask = np.ones_like(original_mask)
mask[:border, :] = False
mask[-border:, :] = False
mask[:, :border] = False
mask[:, -border:] = False
mask = mask & original_mask
```

---

## Analysefehler

### Problem: Analyse liefert nur NaN-Werte

**Mögliche Ursachen:**
1. Zu wenig Kontrast im Bild
2. Zu große Verschiebung
3. Falscher Seed

**Diagnose:**
```python
# Bildkontrast prüfen
ref_gs = ncorr.reference_image.get_gs()
print(f"Kontrast: {ref_gs.std():.4f}")  # Sollte > 0.05 sein

# Seed-Qualität prüfen
for seed in ncorr.seeds:
    print(f"Seed {seed.region_idx}: valid={seed.valid}, "
          f"u={seed.u:.2f}, v={seed.v:.2f}")
```

**Lösungen:**
```python
# 1. Größeren Suchradius für Seeds
ncorr.calculate_seeds(search_radius=100)  # Statt 50

# 2. Step-Analyse für große Verformungen
params.step_analysis.enabled = True

# 3. Manuellen Seed setzen
from ncorr.algorithms.dic import SeedInfo
manual_seed = SeedInfo(x=250, y=250, u=10, v=5, region_idx=0, valid=True)
ncorr._seeds = [manual_seed]
```

### Problem: Schlechte Korrelation (CC < 0.9)

**Mögliche Ursachen:**
1. Beleuchtungsänderung
2. Unschärfe
3. Zu kleine Subsets

**Diagnose:**
```python
# Korrelationsverteilung analysieren
disp = results.displacements[0]
cc = disp.corrcoef[disp.roi]

import matplotlib.pyplot as plt
plt.hist(cc, bins=50)
plt.xlabel('Korrelationskoeffizient')
plt.title(f'Mittelwert: {np.nanmean(cc):.3f}')
plt.show()
```

**Lösungen:**
```python
# Größere Subsets
params = DICParameters(radius=50)  # Statt 30

# Bildvorverarbeitung
from scipy.ndimage import gaussian_filter
ref_gs = gaussian_filter(ref_gs, sigma=0.5)
```

### Problem: Analyse ist sehr langsam

**Ursachen:**
1. Zu viele Analysepunkte
2. Große Bilder
3. Kleine Spacing-Werte

**Lösungen:**
```python
# 1. Größeren Spacing verwenden
params = DICParameters(spacing=10)  # Statt 5

# 2. Lazy Loading für große Bilder
ncorr.set_reference("großes_bild.tif", lazy=True)
ncorr.set_current("verformt.tif", lazy=True)

# 3. ROI auf wichtigen Bereich beschränken
# Statt gesamtes Bild nur relevanten Bereich analysieren
```

---

## Dehnungsprobleme

### Problem: Verrauschte Dehnungsfelder

**Ursache:** Verschiebungsrauschen wird durch Ableitung verstärkt.

**Lösungen:**
```python
# Größeren Dehnungsradius verwenden
from ncorr.algorithms.strain import StrainCalculator
calc = StrainCalculator(strain_radius=10)  # Statt 5

# Verschiebungsfeld vorher glätten
from scipy.ndimage import gaussian_filter
u_smooth = gaussian_filter(disp.u, sigma=2)
v_smooth = gaussian_filter(disp.v, sigma=2)
```

### Problem: Dehnung am Rand nicht berechnet

**Ursache:** Dehnungsradius benötigt Punkte um den Messpunkt.

**Erklärung:**
```
Dehnungsradius = 5 → benötigt 5 Pixel links, rechts, oben, unten
                   → 10 Pixel Rand werden nicht berechnet
```

**Lösung:**
```python
# Kleineren Dehnungsradius am Rand verwenden
# Oder: ROI-Rand in Analyse berücksichtigen
```

---

## Speicherprobleme

### Problem: `MemoryError`

**Lösungen:**
```python
# 1. Lazy Loading verwenden
ncorr.set_reference("bild.tif", lazy=True)

# 2. Größeren Spacing verwenden (weniger Punkte)
params = DICParameters(spacing=10)

# 3. Bilder verkleinern
from ncorr.core.image import NcorrImage
img = NcorrImage.from_file("großes_bild.tif")
img_reduced = img.reduce(spacing=1)  # Halbiert Größe

# 4. Ergebnisse einzeln verarbeiten, nicht alle im Speicher halten
for i, img_path in enumerate(current_images):
    ncorr.set_current(img_path)
    result = ncorr.run_analysis()
    result.save(f"result_{i}")
    del result  # Speicher freigeben
```

---

## Ergebnis-Validierung

### Checkliste für gute Ergebnisse

```python
def validate_results(results):
    """Prüft Ergebnisqualität."""
    issues = []

    d = results.displacements[0]
    s = results.strains_ref[0]

    # 1. Korrelationskoeffizient
    cc = d.corrcoef[d.roi]
    if np.nanmean(cc) < 0.95:
        issues.append(f"Niedrige Korrelation: {np.nanmean(cc):.3f}")

    if np.nanmin(cc) < 0.8:
        issues.append(f"Sehr niedrige Min-Korrelation: {np.nanmin(cc):.3f}")

    # 2. Abdeckung
    coverage = np.sum(d.roi) / d.roi.size * 100
    if coverage < 50:
        issues.append(f"Niedrige Abdeckung: {coverage:.1f}%")

    # 3. Physikalische Plausibilität
    max_strain = np.nanmax(np.abs(s.exx[s.roi]))
    if max_strain > 0.5:  # 50% Dehnung
        issues.append(f"Hohe Dehnung verdächtig: {max_strain*100:.1f}%")

    # 4. Konsistenz
    from scipy.ndimage import laplace
    roughness = np.nanstd(laplace(np.nan_to_num(d.u)))
    if roughness > 1.0:
        issues.append(f"Raues Verschiebungsfeld: {roughness:.2f}")

    if issues:
        print("⚠️ Potenzielle Probleme:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Ergebnisse sehen gut aus")

    return len(issues) == 0

# Verwendung
validate_results(results)
```

---

## Kontakt und Support

Bei Problemen, die hier nicht behandelt werden:

1. **Dokumentation prüfen**: Alle Kapitel durchlesen
2. **Tests ausführen**: `pytest -v` zeigt potenzielle Probleme
3. **Minimal reproduzierbares Beispiel erstellen**
4. **Issue auf GitHub erstellen** mit:
   - Python-Version
   - Ncorr-Version
   - Vollständige Fehlermeldung
   - Minimal reproduzierbarer Code
