# Schnellstart

Dieses Tutorial führt Sie durch eine vollständige DIC-Analyse in wenigen Minuten.

## Vorbereitung

### Benötigte Bilder

Für eine DIC-Analyse benötigen Sie:
1. Ein **Referenzbild** (vor der Verformung)
2. Ein oder mehrere **verformte Bilder** (nach der Verformung)

Die Bilder sollten:
- Ein kontrastreiches Speckle-Muster haben
- Gleiche Größe haben
- Ausreichende Auflösung haben (empfohlen: > 1000 x 1000 Pixel)

### Unterstützte Bildformate

- TIFF (.tif, .tiff) - empfohlen für 16-bit
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)

## Minimales Beispiel

```python
import numpy as np
from ncorr import Ncorr, DICParameters

# 1. Ncorr-Instanz erstellen
ncorr = Ncorr()

# 2. Bilder laden
ncorr.set_reference("referenz.tif")
ncorr.set_current("verformt.tif")

# 3. ROI definieren (hier: gesamtes Bild)
# Erstelle Maske mit Rand von 50 Pixeln
mask = np.ones((1000, 1000), dtype=bool)
mask[:50, :] = False
mask[-50:, :] = False
mask[:, :50] = False
mask[:, -50:] = False
ncorr.set_roi_from_mask(mask)

# 4. Analyse durchführen
results = ncorr.run_analysis()

# 5. Ergebnisse ausgeben
disp = results.displacements[0]
print(f"Mittlere u-Verschiebung: {np.nanmean(disp.u):.3f} Pixel")
print(f"Mittlere v-Verschiebung: {np.nanmean(disp.v):.3f} Pixel")
```

## Vollständiges Beispiel mit Visualisierung

```python
import numpy as np
import matplotlib.pyplot as plt
from ncorr import Ncorr, DICParameters
from ncorr.core.image import NcorrImage

# === 1. Bilder laden ===

# Referenzbild laden
ref_img = NcorrImage.from_file("referenz.tif")
print(f"Referenzbild: {ref_img.width} x {ref_img.height} Pixel")

# Verformtes Bild laden
cur_img = NcorrImage.from_file("verformt.tif")

# === 2. Ncorr initialisieren ===

ncorr = Ncorr()

# Fortschrittsanzeige aktivieren
def progress_callback(progress, message):
    print(f"[{progress*100:.0f}%] {message}")

ncorr.set_progress_callback(progress_callback)

# Bilder setzen
ncorr.set_reference(ref_img)
ncorr.set_current(cur_img)

# === 3. ROI definieren ===

# Kreisförmige ROI im Zentrum
h, w = ref_img.height, ref_img.width
y, x = np.ogrid[:h, :w]
center_x, center_y = w // 2, h // 2
radius = min(w, h) // 3

mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
ncorr.set_roi_from_mask(mask)

# === 4. Parameter konfigurieren ===

params = DICParameters(
    radius=30,              # Subset-Radius (Pixel)
    spacing=5,              # Gitterabstand
    cutoff_diffnorm=1e-3,   # Konvergenztoleranz
    cutoff_iteration=30,    # Maximale Iterationen
)
ncorr.set_parameters(params)

# === 5. Analyse durchführen ===

print("\nStarte DIC-Analyse...")
results = ncorr.run_analysis()
print("Analyse abgeschlossen!")

# === 6. Ergebnisse visualisieren ===

disp = results.displacements[0]
strain_ref = results.strains_ref[0]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Referenzbild
axes[0, 0].imshow(ref_img.get_gs(), cmap='gray')
axes[0, 0].set_title('Referenzbild')
axes[0, 0].axis('off')

# u-Verschiebung
im1 = axes[0, 1].imshow(disp.u, cmap='jet')
axes[0, 1].set_title('u-Verschiebung (Pixel)')
plt.colorbar(im1, ax=axes[0, 1])

# v-Verschiebung
im2 = axes[0, 2].imshow(disp.v, cmap='jet')
axes[0, 2].set_title('v-Verschiebung (Pixel)')
plt.colorbar(im2, ax=axes[0, 2])

# Korrelationskoeffizient
im3 = axes[1, 0].imshow(disp.corrcoef, cmap='hot', vmin=0.9, vmax=1.0)
axes[1, 0].set_title('Korrelationskoeffizient')
plt.colorbar(im3, ax=axes[1, 0])

# εxx Dehnung
im4 = axes[1, 1].imshow(strain_ref.exx, cmap='coolwarm')
axes[1, 1].set_title('εxx (Green-Lagrange)')
plt.colorbar(im4, ax=axes[1, 1])

# εyy Dehnung
im5 = axes[1, 2].imshow(strain_ref.eyy, cmap='coolwarm')
axes[1, 2].set_title('εyy (Green-Lagrange)')
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('dic_ergebnisse.png', dpi=150)
plt.show()

# === 7. Statistiken ausgeben ===

valid = disp.roi
print("\n=== Ergebnisstatistiken ===")
print(f"Analysierte Punkte: {np.sum(valid)}")
print(f"\nVerschiebungen:")
print(f"  u: {np.nanmean(disp.u[valid]):.4f} ± {np.nanstd(disp.u[valid]):.4f} Pixel")
print(f"  v: {np.nanmean(disp.v[valid]):.4f} ± {np.nanstd(disp.v[valid]):.4f} Pixel")
print(f"\nDehnungen:")
print(f"  εxx: {np.nanmean(strain_ref.exx[strain_ref.roi]):.6f}")
print(f"  εyy: {np.nanmean(strain_ref.eyy[strain_ref.roi]):.6f}")
print(f"  εxy: {np.nanmean(strain_ref.exy[strain_ref.roi]):.6f}")
```

## Bildsequenz analysieren

```python
import numpy as np
from ncorr import Ncorr, DICParameters
from pathlib import Path

# Bilddateien finden
image_dir = Path("bilder")
reference = image_dir / "bild_000.tif"
current_images = sorted(image_dir.glob("bild_*.tif"))[1:]  # Alle außer dem ersten

print(f"Referenz: {reference}")
print(f"Verformte Bilder: {len(current_images)}")

# Ncorr initialisieren
ncorr = Ncorr()
ncorr.set_reference(str(reference))
ncorr.set_current([str(f) for f in current_images])

# ROI aus Maskenbild
mask_image = image_dir / "maske.png"
ncorr.set_roi_from_image(str(mask_image), threshold=0.5)

# Parameter
params = DICParameters(
    radius=25,
    spacing=3,
    cutoff_diffnorm=1e-3,
    cutoff_iteration=50,
)
params.step_analysis.enabled = True  # Für große Verformungen
ncorr.set_parameters(params)

# Analyse
results = ncorr.run_analysis()

# Ergebnisse speichern
results.save("analyse_ergebnisse")

# Zeitlicher Verlauf der Verschiebung
u_mean = []
v_mean = []
for i, disp in enumerate(results.displacements):
    valid = disp.roi
    u_mean.append(np.nanmean(disp.u[valid]))
    v_mean.append(np.nanmean(disp.v[valid]))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(u_mean, 'b-o', label='u (horizontal)')
plt.plot(v_mean, 'r-s', label='v (vertikal)')
plt.xlabel('Bildnummer')
plt.ylabel('Mittlere Verschiebung (Pixel)')
plt.legend()
plt.grid(True)
plt.title('Zeitlicher Verschiebungsverlauf')
plt.savefig('verschiebungsverlauf.png')
plt.show()
```

## Ergebnisse exportieren

```python
import numpy as np

# Als NumPy-Arrays speichern
np.save('u_verschiebung.npy', results.displacements[0].u)
np.save('v_verschiebung.npy', results.displacements[0].v)
np.save('epsilon_xx.npy', results.strains_ref[0].exx)

# Als CSV für Excel
import csv

disp = results.displacements[0]
with open('verschiebungen.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y', 'u', 'v', 'korrelation'])

    for y in range(disp.u.shape[0]):
        for x in range(disp.u.shape[1]):
            if disp.roi[y, x]:
                writer.writerow([
                    x, y,
                    disp.u[y, x],
                    disp.v[y, x],
                    disp.corrcoef[y, x]
                ])

# Komplette Ergebnisse als Ncorr-Format
results.save('meine_analyse')  # Erstellt .npz und .json Dateien

# Später laden:
from ncorr.main import AnalysisResults
loaded_results = AnalysisResults.load('meine_analyse')
```

## Nächste Schritte

- [Grundkonzepte](04_grundkonzepte.md) - Verstehen Sie die DIC-Theorie
- [Tutorials](05_tutorials.md) - Detaillierte Anleitungen
- [API-Referenz](06_api_referenz.md) - Vollständige Funktionsreferenz
