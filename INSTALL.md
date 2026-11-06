# Installing and using CalibConsumer

CalibConsumer is pure Python (≥ 3.10) on NumPy. SciPy (re-fitting), matplotlib (plots) and OpenCV (its backend and its
YAML/XML files) are optional. Three ways to use it:

## 1. Straight from the source folder (no installation)

Put `src` on Python's path:

```
git clone https://github.com/Dominik-Src/calibconsumer.git calibconsumer
cd calibconsumer
PYTHONPATH=src python3 -c "import calibconsumer; print(calibconsumer.__version__)"
PYTHONPATH=src python3 -m calibconsumer evaluate --help
```

In your own script, prepend the folder before the import:

```python
import sys
sys.path.insert(0, "/path/to/calibconsumer/src")
import calibconsumer
```

## 2. Installed into an environment (recommended)

A virtual environment keeps it apart from the system's Python:

```
python3 -m venv .venv
. .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install /path/to/calibconsumer         # or, from its folder: pip install .
pip install "/path/to/calibconsumer[plot]" # with matplotlib for the plots
pip install "calibconsumer @ git+https://github.com/Dominik-Src/calibconsumer.git@v0.1.1"   # straight from the repository, a fixed release
```

The extras are `fit` (SciPy), `plot` (matplotlib), `opencv` (opencv-python-headless) and `test` (pytest and the
others, for the tests); several at once as `.[fit,plot]`. The installation brings the command `calibconsumer`:

```
calibconsumer evaluate resultsCalibration/<run>/cameraLensInstance-1 --out evaluation
calibconsumer info mapping.json
```

While changing the package itself, install it editable (`pip install -e ".[test]"`): changes to `src/` apply without
reinstalling.

## 3. As a wheel (to pass on or install offline)

```
pip install build
python3 -m build                           # dist/calibconsumer-<version>-py3-none-any.whl and the .tar.gz
pip install dist/calibconsumer-<version>-py3-none-any.whl
```

The wheel contains no compiled code and runs on every platform.

## Checking the installation

```
pip install ".[test]"
python3 -m pytest -q                       # from the package's folder; about 35 s
python3 examples/reevaluate.py             # writes evaluation-example/ with its plots
```
