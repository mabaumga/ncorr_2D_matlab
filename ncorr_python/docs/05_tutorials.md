# Tutorials

Dieses Kapitel enthält detaillierte Schritt-für-Schritt-Anleitungen für verschiedene Anwendungsfälle.

## Tutorial 1: Zugversuch-Analyse

Analyse eines einfachen Zugversuchs mit konstanter Dehnung.

### Schritt 1: Bilder vorbereiten

```python
import numpy as np
from scipy.ndimage import shift
from PIL import Image

# Synthetisches Speckle-Muster erstellen
def create_speckle_pattern(size=500, seed=42):
    np.random.seed(seed)

    # Basis-Rauschen
    pattern = np.random.rand(size, size)

    # Tiefpass-Filter für realistische Speckles
    from scipy.ndimage import gaussian_filter
    pattern = gaussian_filter(pattern, sigma=3)

    # Normalisieren auf 0-255
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    return (pattern * 255).astype(np.uint8)

# Referenzbild erstellen
ref_pattern = create_speckle_pattern()

# Verformtes Bild (5% Dehnung in x-Richtung)
strain_xx = 0.05
x = np.arange(500)
new_x = x * (1 + strain_xx)

# Interpolation für die Verformung
from scipy.interpolate import interp1d
deformed = np.zeros_like(ref_pattern, dtype=np.float64)
for row in range(500):
    interpolator = interp1d(x, ref_pattern[row, :],
                           kind='cubic',
                           bounds_error=False,
                           fill_value=0)
    deformed[row, :] = interpolator(new_x)

deformed = deformed.astype(np.uint8)

# Bilder speichern
Image.fromarray(ref_pattern).save('zugversuch_ref.tif')
Image.fromarray(deformed).save('zugversuch_def.tif')
```

### Schritt 2: DIC-Analyse durchführen

```python
from ncorr import Ncorr, DICParameters
import numpy as np

# Ncorr initialisieren
ncorr = Ncorr()

# Bilder laden
ncorr.set_reference('zugversuch_ref.tif')
ncorr.set_current('zugversuch_def.tif')

# ROI definieren (zentraler Bereich)
mask = np.zeros((500, 500), dtype=bool)
mask[50:450, 50:450] = True
ncorr.set_roi_from_mask(mask)

# Parameter für Zugversuch
params = DICParameters(
    radius=25,          # Mittelgroßes Subset
    spacing=5,          # Moderate Auflösung
    cutoff_diffnorm=1e-4,
    cutoff_iteration=50,
)
ncorr.set_parameters(params)

# Analyse
results = ncorr.run_analysis()
```

### Schritt 3: Ergebnisse validieren

```python
import matplotlib.pyplot as plt

disp = results.displacements[0]
strain = results.strains_ref[0]

# Erwartete Werte
expected_strain = 0.05
expected_u_max = 500 * 0.05  # = 25 Pixel am rechten Rand

# Tatsächliche Werte
valid = disp.roi
actual_strain = np.nanmean(strain.exx[strain.roi])
actual_u_max = np.nanmax(disp.u[valid])

print(f"Erwartete Dehnung: {expected_strain:.4f}")
print(f"Gemessene Dehnung: {actual_strain:.4f}")
print(f"Abweichung: {abs(expected_strain - actual_strain) / expected_strain * 100:.2f}%")

# Visualisierung
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(disp.u, cmap='jet')
axes[0].set_title('u-Verschiebung (Pixel)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(strain.exx, cmap='coolwarm',
                     vmin=0.04, vmax=0.06)
axes[1].set_title('εxx (sollte 0.05 sein)')
plt.colorbar(im1, ax=axes[1])

# Profil durch die Mitte
mid_row = disp.u.shape[0] // 2
axes[2].plot(disp.u[mid_row, :], 'b-')
axes[2].axhline(y=expected_u_max/2, color='r', linestyle='--',
               label='Erwartet (Mitte)')
axes[2].set_xlabel('x (Pixel)')
axes[2].set_ylabel('u (Pixel)')
axes[2].set_title('u-Profil durch die Mitte')
axes[2].legend()

plt.tight_layout()
plt.savefig('zugversuch_ergebnisse.png', dpi=150)
plt.show()
```

## Tutorial 2: Lochplatte unter Zug

Analyse einer Platte mit Loch unter einachsiger Belastung.

### Schritt 1: Geometrie erstellen

```python
import numpy as np

size = 600
mask = np.ones((size, size), dtype=bool)

# Loch in der Mitte
center = size // 2
hole_radius = 50

y, x = np.ogrid[:size, :size]
hole = ((x - center)**2 + (y - center)**2) <= hole_radius**2

# Maske ohne Loch
mask = ~hole

# Rand ausschließen
border = 60
mask[:border, :] = False
mask[-border:, :] = False
mask[:, :border] = False
mask[:, -border:] = False

# Speichern
from PIL import Image
Image.fromarray((mask * 255).astype(np.uint8)).save('lochplatte_maske.png')
```

### Schritt 2: Analyse

```python
from ncorr import Ncorr, DICParameters

ncorr = Ncorr()
ncorr.set_reference('lochplatte_ref.tif')
ncorr.set_current('lochplatte_def.tif')
ncorr.set_roi_from_image('lochplatte_maske.png')

params = DICParameters(
    radius=20,      # Kleiner für Details am Lochrand
    spacing=3,      # Feineres Gitter
)
ncorr.set_parameters(params)

results = ncorr.run_analysis()
```

### Schritt 3: Spannungskonzentration visualisieren

```python
import matplotlib.pyplot as plt
from ncorr.algorithms.strain import StrainCalculator

strain = results.strains_ref[0]

# Hauptdehnungen berechnen
e1, e2, theta = StrainCalculator.calculate_principal_strains(
    strain.exx, strain.exy, strain.eyy
)

# Von-Mises-Dehnung
e_vm = StrainCalculator.calculate_von_mises(
    strain.exx, strain.exy, strain.eyy
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Maximale Hauptdehnung
im0 = axes[0].imshow(e1, cmap='hot')
axes[0].set_title('Maximale Hauptdehnung ε₁')
plt.colorbar(im0, ax=axes[0])

# Von-Mises-Dehnung
im1 = axes[1].imshow(e_vm, cmap='hot')
axes[1].set_title('Von-Mises-Dehnung')
plt.colorbar(im1, ax=axes[1])

# Dehnung entlang Lochrand
angles = np.linspace(0, 2*np.pi, 100)
r_measure = hole_radius + 5  # Knapp außerhalb des Lochs

x_edge = center + r_measure * np.cos(angles)
y_edge = center + r_measure * np.sin(angles)

strain_at_edge = []
for xe, ye in zip(x_edge, y_edge):
    ix, iy = int(xe), int(ye)
    if 0 <= ix < size and 0 <= iy < size:
        strain_at_edge.append(e1[iy, ix])
    else:
        strain_at_edge.append(np.nan)

axes[2].plot(np.degrees(angles), strain_at_edge)
axes[2].set_xlabel('Winkel (°)')
axes[2].set_ylabel('ε₁ am Lochrand')
axes[2].set_title('Spannungskonzentration')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('lochplatte_ergebnisse.png', dpi=150)
plt.show()

# Spannungskonzentrationsfaktor
nominal_strain = np.nanmean(strain.exx[strain.roi])
max_strain = np.nanmax(e1)
K_t = max_strain / nominal_strain
print(f"Spannungskonzentrationsfaktor K_t ≈ {K_t:.2f}")
```

## Tutorial 3: Zeitaufgelöste Analyse

Analyse einer Bildsequenz zur Verfolgung der Verformung über die Zeit.

### Schritt 1: Bildsequenz laden

```python
from pathlib import Path
from ncorr import Ncorr, DICParameters

# Bilder finden
image_dir = Path('versuchsbilder')
all_images = sorted(image_dir.glob('bild_*.tif'))

print(f"Gefunden: {len(all_images)} Bilder")
print(f"Erstes Bild: {all_images[0].name}")
print(f"Letztes Bild: {all_images[-1].name}")

# Ncorr initialisieren
ncorr = Ncorr()

# Erstes Bild als Referenz
ncorr.set_reference(str(all_images[0]))

# Restliche Bilder als aktuelle Bilder
ncorr.set_current([str(f) for f in all_images[1:]])

# ROI
mask = np.ones((1000, 1000), dtype=bool)
mask[:50, :] = False
mask[-50:, :] = False
mask[:, :50] = False
mask[:, -50:] = False
ncorr.set_roi_from_mask(mask)

# Parameter mit Step-Analyse für große Verformungen
params = DICParameters(
    radius=30,
    spacing=5,
)
params.step_analysis.enabled = True
ncorr.set_parameters(params)

# Fortschrittsanzeige
def show_progress(p, msg):
    print(f"\r[{'='*int(p*50)}{' '*int((1-p)*50)}] {p*100:.0f}% {msg}",
          end='', flush=True)

ncorr.set_progress_callback(show_progress)

# Analyse
print("Starte Analyse...")
results = ncorr.run_analysis()
print("\nFertig!")
```

### Schritt 2: Zeitliche Auswertung

```python
import numpy as np
import matplotlib.pyplot as plt

# Messpunkt definieren (z.B. Probenmitte)
measure_x = 500 // (params.spacing + 1)
measure_y = 500 // (params.spacing + 1)

# Zeitreihen extrahieren
time = np.arange(len(results.displacements))
u_time = []
v_time = []
exx_time = []

for i, (disp, strain) in enumerate(zip(results.displacements,
                                        results.strains_ref)):
    if disp.roi[measure_y, measure_x]:
        u_time.append(disp.u[measure_y, measure_x])
        v_time.append(disp.v[measure_y, measure_x])
        exx_time.append(strain.exx[measure_y, measure_x]
                       if strain.roi[measure_y, measure_x] else np.nan)
    else:
        u_time.append(np.nan)
        v_time.append(np.nan)
        exx_time.append(np.nan)

# Visualisierung
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Verschiebung über Zeit
axes[0, 0].plot(time, u_time, 'b-o', label='u', markersize=3)
axes[0, 0].plot(time, v_time, 'r-s', label='v', markersize=3)
axes[0, 0].set_xlabel('Bildnummer')
axes[0, 0].set_ylabel('Verschiebung (Pixel)')
axes[0, 0].set_title('Verschiebung am Messpunkt')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Dehnung über Zeit
axes[0, 1].plot(time, exx_time, 'g-o', markersize=3)
axes[0, 1].set_xlabel('Bildnummer')
axes[0, 1].set_ylabel('εxx')
axes[0, 1].set_title('Dehnung am Messpunkt')
axes[0, 1].grid(True)

# Verschiebungsfeld zu verschiedenen Zeitpunkten
times_to_show = [0, len(results.displacements)//2, -1]
for i, t in enumerate(times_to_show):
    ax = axes[1, i] if i < 2 else fig.add_subplot(2, 3, 6)
    im = ax.imshow(results.displacements[t].u, cmap='jet')
    ax.set_title(f't = {t if t >= 0 else len(results.displacements)+t}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('zeitanalyse.png', dpi=150)
plt.show()
```

### Schritt 3: Animation erstellen

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(8, 8))

# Erste Verschiebung
im = ax.imshow(results.displacements[0].u, cmap='jet',
               vmin=np.nanmin([d.u for d in results.displacements]),
               vmax=np.nanmax([d.u for d in results.displacements]))
plt.colorbar(im, label='u (Pixel)')
title = ax.set_title('Bild 0')

def update(frame):
    im.set_array(results.displacements[frame].u)
    title.set_text(f'Bild {frame}')
    return [im, title]

ani = animation.FuncAnimation(
    fig, update, frames=len(results.displacements),
    interval=100, blit=True
)

# Als GIF speichern
ani.save('verschiebung_animation.gif', writer='pillow', fps=10)

# Als MP4 speichern (benötigt ffmpeg)
# ani.save('verschiebung_animation.mp4', writer='ffmpeg', fps=30)

plt.show()
```

## Tutorial 4: Bruchmechanik - Rissspitzenanalyse

### Schritt 1: ROI um Rissspitze

```python
import numpy as np
from ncorr import Ncorr, DICParameters

# Angenommen: Rissspitze bei (300, 250)
crack_tip = (300, 250)

# Kreisförmige ROI um Rissspitze
size = 500
y, x = np.ogrid[:size, :size]
r = 100  # Radius um Rissspitze

mask = ((x - crack_tip[0])**2 + (y - crack_tip[1])**2) <= r**2

# Riss ausschließen (horizontaler Riss von links kommend)
crack_width = 5
crack = (np.abs(y - crack_tip[1]) < crack_width) & (x < crack_tip[0])
mask = mask & ~crack

# Ncorr konfigurieren
ncorr = Ncorr()
ncorr.set_reference('riss_ref.tif')
ncorr.set_current('riss_def.tif')
ncorr.set_roi_from_mask(mask)

# Kleine Subsets für hohe Auflösung nahe der Rissspitze
params = DICParameters(
    radius=15,
    spacing=2,
    subset_trunc=True,  # Wichtig für Risse!
)
ncorr.set_parameters(params)

results = ncorr.run_analysis()
```

### Schritt 2: Rissöffnungsverschiebung (COD)

```python
disp = results.displacements[0]

# Punkte oberhalb und unterhalb des Risses
y_above = crack_tip[1] - 20
y_below = crack_tip[1] + 20
x_range = range(crack_tip[0] - 50, crack_tip[0], 2)

cod = []
x_positions = []

for x_pos in x_range:
    # In reduzierte Koordinaten umrechnen
    rx = x_pos // (params.spacing + 1)
    ry_above = y_above // (params.spacing + 1)
    ry_below = y_below // (params.spacing + 1)

    if (0 <= rx < disp.v.shape[1] and
        0 <= ry_above < disp.v.shape[0] and
        0 <= ry_below < disp.v.shape[0]):

        v_above = disp.v[ry_above, rx]
        v_below = disp.v[ry_below, rx]

        if not np.isnan(v_above) and not np.isnan(v_below):
            cod.append(v_above - v_below)
            x_positions.append(x_pos)

# Visualisierung
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Verschiebungsfeld
im = axes[0].imshow(disp.v, cmap='coolwarm')
axes[0].plot(crack_tip[0], crack_tip[1], 'r*', markersize=15,
            label='Rissspitze')
axes[0].set_title('v-Verschiebung')
plt.colorbar(im, ax=axes[0])

# COD-Profil
axes[1].plot(x_positions, cod, 'b-o')
axes[1].axhline(y=0, color='k', linestyle='--')
axes[1].set_xlabel('x (Pixel)')
axes[1].set_ylabel('COD (Pixel)')
axes[1].set_title('Rissöffnungsverschiebung')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## Tutorial 5: Qualitätskontrolle

### Korrelationskoeffizienten prüfen

```python
disp = results.displacements[0]

# Statistik
valid = disp.roi
cc = disp.corrcoef[valid]

print("Korrelationskoeffizient:")
print(f"  Minimum: {np.nanmin(cc):.4f}")
print(f"  Maximum: {np.nanmax(cc):.4f}")
print(f"  Mittelwert: {np.nanmean(cc):.4f}")
print(f"  Std. Abw.: {np.nanstd(cc):.4f}")

# Schwache Korrelation finden
threshold = 0.9
weak = cc < threshold
print(f"\nPunkte mit CC < {threshold}: {np.sum(weak)} ({100*np.sum(weak)/len(cc):.1f}%)")

# Visualisierung
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Korrelationskoeffizient
im = axes[0].imshow(disp.corrcoef, cmap='hot', vmin=0.85, vmax=1.0)
axes[0].set_title('Korrelationskoeffizient')
plt.colorbar(im, ax=axes[0])

# Histogramm
axes[1].hist(cc, bins=50, edgecolor='black')
axes[1].axvline(x=threshold, color='r', linestyle='--', label=f'Schwelle ({threshold})')
axes[1].set_xlabel('Korrelationskoeffizient')
axes[1].set_ylabel('Anzahl')
axes[1].set_title('Verteilung')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Verschiebungsrauschen messen

```python
# Bei einem unverformten Bildpaar sollte die Verschiebung null sein
# Verwenden Sie zwei identische oder nahezu identische Bilder

# Rauschen im Nulltest
u_noise = disp.u[valid]
v_noise = disp.v[valid]

print("Verschiebungsrauschen (sollte nahe 0 sein):")
print(f"  u: {np.nanstd(u_noise):.4f} Pixel RMS")
print(f"  v: {np.nanstd(v_noise):.4f} Pixel RMS")

# Räumliche Gradientenprüfung
from scipy.ndimage import laplace

laplacian_u = laplace(disp.u)
laplacian_v = laplace(disp.v)

print(f"\nRäumliche Konsistenz (Laplace):")
print(f"  u: {np.nanstd(laplacian_u[valid]):.4f}")
print(f"  v: {np.nanstd(laplacian_v[valid]):.4f}")
```
