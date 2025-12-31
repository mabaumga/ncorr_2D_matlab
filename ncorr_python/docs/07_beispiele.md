# Anwendungsbeispiele

Praktische Codebeispiele für verschiedene Anwendungsfälle.

## Beispiel 1: Minimale DIC-Analyse

```python
"""
Minimales Beispiel für eine DIC-Analyse.
"""

import numpy as np
from ncorr import Ncorr, DICParameters

# Ncorr initialisieren
ncorr = Ncorr()

# Bilder laden
ncorr.set_reference("ref.tif")
ncorr.set_current("def.tif")

# Einfache rechteckige ROI
height, width = 1000, 1000  # Anpassen an Ihre Bildgröße
mask = np.ones((height, width), dtype=bool)
mask[:50, :] = mask[-50:, :] = mask[:, :50] = mask[:, -50:] = False
ncorr.set_roi_from_mask(mask)

# Standard-Parameter
ncorr.set_parameters(DICParameters())

# Analyse
results = ncorr.run_analysis()

# Ausgabe
d = results.displacements[0]
print(f"u: {np.nanmean(d.u[d.roi]):.3f} ± {np.nanstd(d.u[d.roi]):.3f} px")
print(f"v: {np.nanmean(d.v[d.roi]):.3f} ± {np.nanstd(d.v[d.roi]):.3f} px")
```

## Beispiel 2: Parameteroptimierung

```python
"""
Systematische Parameteroptimierung für beste Ergebnisse.
"""

import numpy as np
from ncorr import Ncorr, DICParameters

def analyze_with_params(ref_path, cur_path, mask, params):
    """Führt Analyse mit gegebenen Parametern durch."""
    ncorr = Ncorr()
    ncorr.set_reference(ref_path)
    ncorr.set_current(cur_path)
    ncorr.set_roi_from_mask(mask)
    ncorr.set_parameters(params)
    return ncorr.run_analysis()

def evaluate_quality(results):
    """Bewertet die Ergebnisqualität."""
    d = results.displacements[0]
    valid = d.roi

    # Qualitätsmetriken
    mean_cc = np.nanmean(d.corrcoef[valid])
    valid_ratio = np.sum(valid) / valid.size

    # Glattheit der Verschiebungsfelder
    from scipy.ndimage import laplace
    laplacian_u = np.nanstd(laplace(np.nan_to_num(d.u)))

    return {
        'mean_correlation': mean_cc,
        'valid_ratio': valid_ratio,
        'smoothness': 1 / (1 + laplacian_u),
    }

# Parameterkombinationen testen
radii = [20, 30, 40]
spacings = [3, 5, 7]

results_table = []

for radius in radii:
    for spacing in spacings:
        params = DICParameters(radius=radius, spacing=spacing)
        results = analyze_with_params("ref.tif", "def.tif", mask, params)
        quality = evaluate_quality(results)

        results_table.append({
            'radius': radius,
            'spacing': spacing,
            **quality
        })

        print(f"Radius={radius}, Spacing={spacing}: "
              f"CC={quality['mean_correlation']:.4f}, "
              f"Valid={quality['valid_ratio']*100:.1f}%")

# Beste Parameter finden
best = max(results_table, key=lambda x: x['mean_correlation'])
print(f"\nBeste Parameter: Radius={best['radius']}, Spacing={best['spacing']}")
```

## Beispiel 3: Batch-Verarbeitung

```python
"""
Automatische Verarbeitung vieler Versuche.
"""

import numpy as np
from pathlib import Path
import json
from ncorr import Ncorr, DICParameters

def process_experiment(experiment_dir):
    """Verarbeitet einen Versuch."""
    exp_dir = Path(experiment_dir)

    # Dateien finden
    ref_img = exp_dir / "reference.tif"
    cur_imgs = sorted(exp_dir.glob("deformed_*.tif"))
    mask_img = exp_dir / "mask.png"

    if not ref_img.exists():
        return None, f"Referenzbild nicht gefunden: {ref_img}"

    if not cur_imgs:
        return None, "Keine verformten Bilder gefunden"

    # Analyse
    ncorr = Ncorr()
    ncorr.set_reference(str(ref_img))
    ncorr.set_current([str(f) for f in cur_imgs])

    if mask_img.exists():
        ncorr.set_roi_from_image(str(mask_img))
    else:
        # Standard-ROI
        from PIL import Image
        with Image.open(ref_img) as img:
            h, w = img.size[1], img.size[0]
        mask = np.ones((h, w), dtype=bool)
        mask[:50, :] = mask[-50:, :] = mask[:, :50] = mask[:, -50:] = False
        ncorr.set_roi_from_mask(mask)

    params = DICParameters(radius=30, spacing=5)
    ncorr.set_parameters(params)

    try:
        results = ncorr.run_analysis()
        results.save(exp_dir / "results")
        return results, "Erfolgreich"
    except Exception as e:
        return None, str(e)

def batch_process(base_dir, experiments):
    """Verarbeitet mehrere Versuche."""
    base = Path(base_dir)
    summary = []

    for exp_name in experiments:
        print(f"\nVerarbeite: {exp_name}")
        exp_dir = base / exp_name

        results, message = process_experiment(exp_dir)

        status = "OK" if results else "FEHLER"
        summary.append({
            'experiment': exp_name,
            'status': status,
            'message': message,
        })

        if results:
            # Zusammenfassung extrahieren
            d = results.displacements[-1]
            valid = d.roi
            summary[-1]['max_u'] = float(np.nanmax(np.abs(d.u[valid])))
            summary[-1]['max_v'] = float(np.nanmax(np.abs(d.v[valid])))

    # Zusammenfassung speichern
    with open(base / "batch_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary

# Verwendung
experiments = ["test_001", "test_002", "test_003"]
summary = batch_process("/pfad/zu/versuchen", experiments)

for s in summary:
    print(f"{s['experiment']}: {s['status']} - {s['message']}")
```

## Beispiel 4: Echtzeit-Überwachung

```python
"""
Pseudo-Echtzeit DIC für Monitoring-Anwendungen.
"""

import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from ncorr import Ncorr, DICParameters

class RealtimeDIC:
    def __init__(self, ref_path, mask, params):
        self.ncorr = Ncorr()
        self.ncorr.set_reference(ref_path)
        self.ncorr.set_roi_from_mask(mask)
        self.ncorr.set_parameters(params)

        self.history = {
            'time': [],
            'u_mean': [],
            'v_mean': [],
            'exx_max': [],
        }

        # Plot initialisieren
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))

    def process_new_image(self, image_path, timestamp=None):
        """Verarbeitet ein neues Bild."""
        if timestamp is None:
            timestamp = time.time()

        self.ncorr.set_current(image_path)
        results = self.ncorr.run_analysis()

        d = results.displacements[0]
        s = results.strains_ref[0]
        valid = d.roi

        # Historie aktualisieren
        self.history['time'].append(timestamp)
        self.history['u_mean'].append(np.nanmean(d.u[valid]))
        self.history['v_mean'].append(np.nanmean(d.v[valid]))
        self.history['exx_max'].append(np.nanmax(s.exx[s.roi]) if np.any(s.roi) else np.nan)

        # Plot aktualisieren
        self.update_plot(d, s)

        return results

    def update_plot(self, d, s):
        """Aktualisiert die Anzeige."""
        for ax in self.axes.flat:
            ax.clear()

        # Aktuelles Verschiebungsfeld
        im0 = self.axes[0, 0].imshow(d.u, cmap='jet')
        self.axes[0, 0].set_title('u-Verschiebung')
        plt.colorbar(im0, ax=self.axes[0, 0])

        im1 = self.axes[0, 1].imshow(s.exx, cmap='coolwarm')
        self.axes[0, 1].set_title('εxx')
        plt.colorbar(im1, ax=self.axes[0, 1])

        # Zeitverläufe
        t = self.history['time']
        t_rel = [ti - t[0] for ti in t]

        self.axes[1, 0].plot(t_rel, self.history['u_mean'], 'b-')
        self.axes[1, 0].set_xlabel('Zeit (s)')
        self.axes[1, 0].set_ylabel('Mittlere u-Verschiebung')
        self.axes[1, 0].grid(True)

        self.axes[1, 1].plot(t_rel, self.history['exx_max'], 'r-')
        self.axes[1, 1].set_xlabel('Zeit (s)')
        self.axes[1, 1].set_ylabel('Max. εxx')
        self.axes[1, 1].grid(True)

        plt.tight_layout()
        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.close()

# Verwendung
mask = np.ones((500, 500), dtype=bool)
mask[:30, :] = mask[-30:, :] = mask[:, :30] = mask[:, -30:] = False

params = DICParameters(radius=20, spacing=5)

monitor = RealtimeDIC("reference.tif", mask, params)

# Simuliere Echtzeit-Bilder
image_dir = Path("live_images")
for img_path in sorted(image_dir.glob("frame_*.tif")):
    print(f"Verarbeite: {img_path.name}")
    monitor.process_new_image(str(img_path))
    time.sleep(0.1)  # Wartezeit zwischen Bildern

monitor.close()
```

## Beispiel 5: Export für FEM-Vergleich

```python
"""
Export von DIC-Ergebnissen für FEM-Vergleich.
"""

import numpy as np

def export_for_fem(results, spacing, output_prefix):
    """
    Exportiert DIC-Ergebnisse im Format für FEM-Vergleich.
    """
    d = results.displacements[0]
    s = results.strains_ref[0]

    # Koordinatengitter erstellen
    h, w = d.u.shape
    step = spacing + 1

    x = np.arange(w) * step
    y = np.arange(h) * step
    X, Y = np.meshgrid(x, y)

    # Nur gültige Punkte
    valid = d.roi & s.roi

    # Als strukturiertes Array
    n_points = np.sum(valid)
    data = np.zeros(n_points, dtype=[
        ('x', 'f8'),
        ('y', 'f8'),
        ('u', 'f8'),
        ('v', 'f8'),
        ('exx', 'f8'),
        ('eyy', 'f8'),
        ('exy', 'f8'),
    ])

    data['x'] = X[valid]
    data['y'] = Y[valid]
    data['u'] = d.u[valid]
    data['v'] = d.v[valid]
    data['exx'] = s.exx[valid]
    data['eyy'] = s.eyy[valid]
    data['exy'] = s.exy[valid]

    # Als CSV speichern
    header = "x,y,u,v,exx,eyy,exy"
    np.savetxt(
        f"{output_prefix}.csv",
        np.column_stack([data[f] for f in data.dtype.names]),
        delimiter=',',
        header=header,
        comments='',
        fmt='%.6f'
    )

    # Als VTK für ParaView
    try:
        export_vtk(X, Y, d.u, d.v, s.exx, s.eyy, s.exy, valid, f"{output_prefix}.vtk")
    except ImportError:
        print("VTK-Export übersprungen (pyevtk nicht installiert)")

    return data

def export_vtk(X, Y, u, v, exx, eyy, exy, valid, filename):
    """Export als VTK-Datei für ParaView."""
    from pyevtk.hl import gridToVTK

    # Gitterdaten vorbereiten
    nx, ny = X.shape[1], X.shape[0]
    x = np.ascontiguousarray(X[0, :])
    y = np.ascontiguousarray(Y[:, 0])
    z = np.array([0.0])

    # Felder vorbereiten (NaN durch 0 ersetzen)
    u_data = np.nan_to_num(u).reshape((ny, nx, 1)).astype(np.float64)
    v_data = np.nan_to_num(v).reshape((ny, nx, 1)).astype(np.float64)
    exx_data = np.nan_to_num(exx).reshape((ny, nx, 1)).astype(np.float64)
    eyy_data = np.nan_to_num(eyy).reshape((ny, nx, 1)).astype(np.float64)
    exy_data = np.nan_to_num(exy).reshape((ny, nx, 1)).astype(np.float64)

    gridToVTK(
        filename.replace('.vtk', ''),
        x, y, z,
        pointData={
            "u": np.ascontiguousarray(u_data),
            "v": np.ascontiguousarray(v_data),
            "exx": np.ascontiguousarray(exx_data),
            "eyy": np.ascontiguousarray(eyy_data),
            "exy": np.ascontiguousarray(exy_data),
        }
    )

# Verwendung
data = export_for_fem(results, params.spacing, "fem_vergleich")
print(f"Exportiert: {len(data)} Datenpunkte")
```

## Beispiel 6: Virtuelle Dehnungsmessstreifen

```python
"""
Berechnung von virtuellen Dehnungsmessstreifen (DMS) aus DIC-Daten.
"""

import numpy as np
import matplotlib.pyplot as plt

def virtual_strain_gauge(results, x_start, y_start, x_end, y_end, gauge_length_mm, pix_to_mm):
    """
    Berechnet virtuelle DMS-Messung entlang einer Linie.

    Parameter:
        x_start, y_start: Startpunkt (Pixel)
        x_end, y_end: Endpunkt (Pixel)
        gauge_length_mm: Nominale DMS-Länge (mm)
        pix_to_mm: Pixel-zu-mm Umrechnung
    """
    d = results.displacements[0]
    spacing = results.parameters.spacing + 1

    # Punkte entlang der Linie
    n_points = 50
    t = np.linspace(0, 1, n_points)
    x_line = x_start + t * (x_end - x_start)
    y_line = y_start + t * (y_end - y_start)

    # In reduzierte Koordinaten
    x_idx = (x_line / spacing).astype(int)
    y_idx = (y_line / spacing).astype(int)

    # Verschiebungen entlang der Linie
    u_line = []
    v_line = []

    for xi, yi in zip(x_idx, y_idx):
        if 0 <= yi < d.u.shape[0] and 0 <= xi < d.u.shape[1]:
            u_line.append(d.u[yi, xi])
            v_line.append(d.v[yi, xi])
        else:
            u_line.append(np.nan)
            v_line.append(np.nan)

    u_line = np.array(u_line)
    v_line = np.array(v_line)

    # Länge der Linie
    line_length_pix = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    line_length_mm = line_length_pix * pix_to_mm

    # Richtungsvektor
    dx = (x_end - x_start) / line_length_pix
    dy = (y_end - y_start) / line_length_pix

    # Verschiebung in Linienrichtung
    u_axial = u_line * dx + v_line * dy

    # Dehnung berechnen (Differenz Start/Ende bezogen auf Länge)
    valid = ~np.isnan(u_axial)
    if np.sum(valid) < 2:
        return None

    # Lineare Regression für robustere Schätzung
    from scipy import stats
    s_valid = np.linspace(0, line_length_mm, n_points)[valid]
    u_valid = u_axial[valid] * pix_to_mm

    slope, intercept, r, p, se = stats.linregress(s_valid, u_valid)

    strain = slope  # du/ds = Dehnung

    return {
        'strain': strain,
        'start': (x_start, y_start),
        'end': (x_end, y_end),
        'length_mm': line_length_mm,
        'r_squared': r**2,
    }

# Verwendung
pix_to_mm = 0.05  # 50 µm/Pixel

# Drei virtuelle DMS definieren
gauges = [
    {'name': 'DMS1', 'x1': 100, 'y1': 250, 'x2': 200, 'y2': 250},
    {'name': 'DMS2', 'x1': 250, 'y1': 100, 'x2': 250, 'y2': 200},
    {'name': 'DMS3', 'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400},
]

print("Virtuelle DMS-Ergebnisse:")
print("-" * 50)

for g in gauges:
    result = virtual_strain_gauge(
        results,
        g['x1'], g['y1'], g['x2'], g['y2'],
        gauge_length_mm=5.0,
        pix_to_mm=pix_to_mm
    )

    if result:
        print(f"{g['name']}: ε = {result['strain']*100:.4f}% "
              f"(R² = {result['r_squared']:.4f})")

# Visualisierung
plt.figure(figsize=(10, 8))
plt.imshow(results.displacements[0].u, cmap='jet')
plt.colorbar(label='u (Pixel)')

for g in gauges:
    plt.plot([g['x1'], g['x2']], [g['y1'], g['y2']], 'w-', linewidth=3)
    plt.plot([g['x1'], g['x2']], [g['y1'], g['y2']], 'k--', linewidth=1)
    mid_x = (g['x1'] + g['x2']) / 2
    mid_y = (g['y1'] + g['y2']) / 2
    plt.text(mid_x, mid_y, g['name'], color='white', fontsize=12,
             ha='center', va='bottom')

plt.title('Positionen der virtuellen Dehnungsmessstreifen')
plt.savefig('virtuelle_dms.png', dpi=150)
plt.show()
```

## Beispiel 7: Fehleranalyse und Unsicherheitsquantifizierung

```python
"""
Unsicherheitsanalyse für DIC-Messungen.
"""

import numpy as np
from scipy import stats

def uncertainty_analysis(ref_path, mask, params, n_trials=10):
    """
    Schätzt die Messunsicherheit durch wiederholte Messungen.
    """
    from ncorr import Ncorr

    results_list = []

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}")

        ncorr = Ncorr()
        ncorr.set_reference(ref_path)
        ncorr.set_current(ref_path)  # Gleiches Bild = sollte 0 sein
        ncorr.set_roi_from_mask(mask)
        ncorr.set_parameters(params)

        result = ncorr.run_analysis()
        results_list.append(result)

    # Statistik berechnen
    all_u = np.array([r.displacements[0].u for r in results_list])
    all_v = np.array([r.displacements[0].v for r in results_list])

    # Bias (systematischer Fehler)
    u_bias = np.nanmean(all_u, axis=0)
    v_bias = np.nanmean(all_v, axis=0)

    # Standardabweichung (zufälliger Fehler)
    u_std = np.nanstd(all_u, axis=0)
    v_std = np.nanstd(all_v, axis=0)

    # Zusammenfassung
    valid = results_list[0].displacements[0].roi

    print("\n=== Unsicherheitsanalyse ===")
    print(f"Trials: {n_trials}")
    print(f"\nBias (systematischer Fehler):")
    print(f"  u: {np.nanmean(u_bias[valid]):.6f} ± {np.nanstd(u_bias[valid]):.6f} px")
    print(f"  v: {np.nanmean(v_bias[valid]):.6f} ± {np.nanstd(v_bias[valid]):.6f} px")
    print(f"\nPräzision (zufälliger Fehler, 1σ):")
    print(f"  u: {np.nanmean(u_std[valid]):.6f} px")
    print(f"  v: {np.nanmean(v_std[valid]):.6f} px")

    # 95% Konfidenzintervall
    u_95 = stats.t.ppf(0.975, n_trials-1) * np.nanmean(u_std[valid]) / np.sqrt(n_trials)
    v_95 = stats.t.ppf(0.975, n_trials-1) * np.nanmean(v_std[valid]) / np.sqrt(n_trials)
    print(f"\n95% Konfidenzintervall:")
    print(f"  u: ±{u_95:.6f} px")
    print(f"  v: ±{v_95:.6f} px")

    return {
        'bias_u': u_bias,
        'bias_v': v_bias,
        'std_u': u_std,
        'std_v': v_std,
        'results': results_list,
    }

# Verwendung
mask = np.ones((500, 500), dtype=bool)
mask[:50, :] = mask[-50:, :] = mask[:, :50] = mask[:, -50:] = False

from ncorr import DICParameters
params = DICParameters(radius=30, spacing=5)

uncertainty = uncertainty_analysis("referenz.tif", mask, params, n_trials=5)
```
