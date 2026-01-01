# Installation

## Voraussetzungen

### Python-Version
Ncorr benötigt Python 3.9 oder höher. Überprüfen Sie Ihre Version:

```bash
python --version
# oder
python3 --version
```

### Virtuelle Umgebung (empfohlen)

Es wird empfohlen, eine virtuelle Umgebung zu verwenden:

```bash
# Erstellen der virtuellen Umgebung
python -m venv ncorr_env

# Aktivieren (Linux/macOS)
source ncorr_env/bin/activate

# Aktivieren (Windows)
ncorr_env\Scripts\activate
```

## Installation von Ncorr

### Option 1: Aus dem Quellcode

```bash
# Repository klonen oder Verzeichnis kopieren
cd ncorr_python

# Entwicklungsinstallation
pip install -e .

# Mit Entwicklungswerkzeugen
pip install -e ".[dev]"
```

### Option 2: Nur Abhängigkeiten installieren

Falls Sie das Paket nicht installieren möchten:

```bash
pip install numpy scipy pillow numba opencv-python matplotlib
```

Dann fügen Sie den Pfad zu Ihrem Python-Skript hinzu:

```python
import sys
sys.path.insert(0, '/pfad/zu/ncorr_python')

from ncorr import Ncorr
```

## Abhängigkeiten

### Erforderliche Pakete

| Paket | Version | Zweck |
|-------|---------|-------|
| numpy | >= 1.21.0 | Numerische Berechnungen |
| scipy | >= 1.7.0 | Wissenschaftliche Funktionen |
| pillow | >= 9.0.0 | Bildverarbeitung |
| numba | >= 0.55.0 | JIT-Kompilierung |
| opencv-python | >= 4.5.0 | Bildverarbeitung |
| matplotlib | >= 3.5.0 | Visualisierung |

### Optionale Pakete

| Paket | Zweck |
|-------|-------|
| PyQt6 | GUI-Unterstützung |
| pytest | Tests ausführen |
| black | Code-Formatierung |

## Installation überprüfen

Führen Sie folgenden Code aus, um die Installation zu testen:

```python
# test_installation.py
from ncorr import Ncorr, DICParameters, NcorrImage
from ncorr.core.status import Status

print("Ncorr erfolgreich importiert!")

# Erstelle ein Testobjekt
ncorr = Ncorr()
print(f"Ncorr-Instanz erstellt: {ncorr}")

# Erstelle Testparameter
params = DICParameters(radius=30, spacing=5)
print(f"Parameter: radius={params.radius}, spacing={params.spacing}")

print("\nInstallation erfolgreich!")
```

Speichern und ausführen:

```bash
python test_installation.py
```

Erwartete Ausgabe:
```
Ncorr erfolgreich importiert!
Ncorr-Instanz erstellt: <ncorr.main.Ncorr object at 0x...>
Parameter: radius=30, spacing=5

Installation erfolgreich!
```

## Tests ausführen

Wenn Sie die Entwicklungsabhängigkeiten installiert haben:

```bash
# Alle Tests
pytest

# Mit Details
pytest -v

# Mit Coverage
pytest --cov=ncorr

# Nur bestimmte Tests
pytest tests/test_core.py -v
```

## Fehlerbehebung bei der Installation

### Numba-Kompilierungsfehler

Falls Numba-Fehler auftreten:

```bash
# Numba-Cache löschen
python -c "import numba; numba.config.CACHE_DIR = None"

# Oder Cache-Verzeichnis manuell löschen
rm -rf ~/.cache/numba
```

### OpenCV-Probleme

Bei Headless-Systemen:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Windows-spezifische Probleme

Falls Kompilierungsfehler auftreten:

```bash
# Visual C++ Build Tools installieren
# Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Oder Anaconda verwenden
conda install numpy scipy numba pillow
```

## Nächste Schritte

Nach erfolgreicher Installation:
- [Schnellstart](03_schnellstart.md) - Erste Analyse durchführen
