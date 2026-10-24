# SPECTRALIS VOL Analyzer

A research-oriented Python pipeline for exploring and analysing **Heidelberg Engineering SPECTRALIS `.vol` OCT files**.

The pipeline loads OCT volumes with [EyePy](https://github.com/MedVisBonn/eyepy), inspects the contents of each file, exports raw B-scans and metadata, evaluates available retinal-layer segmentations, calculates physical scan dimensions, generates retinal thickness maps and en-face projections, localises the foveal pit, performs ETDRS-based macular measurements, and saves both visual and machine-readable outputs.

The main goal is to provide a **transparent, reproducible starting point for OCT research**, particularly for studies involving retinal morphology, macular thickness, longitudinal imaging, and myopia.

**Repository status:** The Python implementation described below has not yet been added to this repository. The installation and usage examples will be runnable once `spectralis_vol_analyzer.py` is committed.

---

## Overview

A Heidelberg `.vol` file can contain considerably more than a set of OCT images.

Depending on the acquisition and export, it may include:

- a 3D OCT volume;
- individual B-scans;
- retinal-layer segmentations;
- NIR/SLO localizer imaging;
- physical pixel spacing;
- B-scan acquisition coordinates;
- image-quality information;
- laterality and acquisition metadata;
- predefined retinal slab definitions.

This pipeline extracts and organises these components and derives additional measurements from them.

The processing workflow is approximately:

```text
Heidelberg .vol
      │
      ▼
Load with EyePy
      │
      ├── OCT volume
      ├── NIR localizer
      ├── metadata
      ├── B-scan geometry
      ├── acquisition quality
      └── retinal segmentations
      │
      ▼
Physical geometry / calibration checks
      │
      ▼
ILM–BM retinal thickness map
      │
      ├── fovea localisation
      ├── ETDRS analysis
      ├── NIR thickness overlay
      ├── layer-depth maps
      └── depth projections
      │
      ▼
Plots + numerical data + reports
```

---

## Features

### 1. Batch processing of Heidelberg `.vol` files

The script automatically finds every `.vol` file in an input directory and processes them independently.

A corrupted or incompatible file does not stop the complete batch.

```python
results = run_batch()
```

Recursive folder searching can optionally be enabled.

---

### 2. OCT volume inspection

For each file, the pipeline reports the OCT volume shape as:

```text
(B-scans, depth samples, A-scans)
```

For example:

```text
(25, 496, 1024)
```

means:

```text
25 B-scans
496 axial/depth samples per A-scan
1024 A-scans per B-scan
```

Conceptually:

```text
A-scan
│
│  depth reflectivity profile
▼

Many A-scans side-by-side
        ↓
      B-scan

Many B-scans acquired across the retina
        ↓
     OCT volume
```

---

## Physical OCT Geometry

The pipeline reads the physical spacing supplied by the Heidelberg file:

```text
scale_x → lateral spacing between A-scans
scale_y → axial/depth spacing
scale_z → spacing between B-scans
```

These values are used to calculate the physical dimensions of the scan.

For an image dimension containing $N$ samples with spacing $s$, two related quantities are reported:

```math
\text{pixel-covered extent}=N\,s
```

and

```math
\text{first-to-last sample distance}=(N-1)s
```

The pipeline also independently checks the stored B-scan coordinates against the header spacing and reports substantial disagreements.

---

## B-scan Export and Visualisation

Every B-scan can be exported at its native resolution as a lossless PNG.

The pipeline also generates:

#### Representative B-scans

```text
First B-scan
Middle B-scan
Last B-scan
```

#### Complete B-scan grids

All B-scans are displayed in paginated overview figures.

This allows rapid visual inspection of:

- macular anatomy;
- scan coverage;
- foveal morphology;
- artefacts;
- segmentation errors;
- signal-quality variation.

---

## NIR / SLO Localizer

If available, the native NIR localizer is exported.

The physical positions of the B-scans are plotted directly on the localizer using the scan coordinates stored in the `.vol` file.

This helps verify:

- scan orientation;
- retinal coverage;
- B-scan ordering;
- foveal position;
- registration between OCT-derived maps and the localizer.

The Heidelberg convention that **B-scan index 0 is commonly the inferior-most scan** is explicitly handled when displaying en-face maps.

---

## Retinal Layer Segmentation

The pipeline detects which retinal boundaries actually contain valid segmentation data.

For every available layer it reports:

```text
layer name
array shape
valid coverage
```

Example:

```text
ILM      shape=(25, 1024)   coverage=99.0%
BM       shape=(25, 1024)   coverage=98.4%
```

Coverage is calculated as:

```math
\text{coverage}
=
\frac{\text{number of finite segmentation values}}
     {\text{total number of A-scan positions}}
\times 100
```

Layer slots containing only missing values are ignored.

---

## Segmentation Overlay

Available retinal boundaries can be plotted directly over a B-scan.

For example:

```text
ILM
BM
```

This provides a visual QC step before thickness measurements are interpreted.

---

## Retinal Thickness Map

When both the **inner limiting membrane (ILM)** and **Bruch's membrane (BM)** are available, total retinal thickness is calculated at every A-scan.

For B-scan $b$ and A-scan $a$:

```math
T(b,a)
=
\left[
BM(b,a)-ILM(b,a)
\right]
\times scale_y
\times 1000
```

where:

```text
b       = B-scan index
a       = A-scan index
ILM     = ILM depth in pixels
BM      = BM depth in pixels
scale_y = axial spacing in mm/pixel
1000    = conversion from mm to µm
```

The resulting array has dimensions:

```text
n_B_scans × n_A_scans
```

and represents an en-face retinal thickness map.

Invalid or non-positive thickness values are stored as `NaN` rather than being artificially filled.

---

## Fovea Localisation

### Current approach

The current implementation defines the foveal-pit candidate as the **minimum of the smoothed ILM-to-BM retinal thickness map**.

Importantly, the algorithm does **not** assume that the fovea is located in:

```text
the middle B-scan
or
the middle A-scan
```

Instead, it searches across the acquired macular volume.

#### Algorithm

```text
ILM + BM segmentation
          │
          ▼
Retinal thickness map
          │
          ▼
NaN-aware Gaussian smoothing
          │
          ▼
Exclude scan-edge margin
          │
          ▼
Find minimum retinal thickness
          │
          ▼
Foveal-pit candidate
          │
          ▼
Convert to physical and NIR coordinates
          │
          ▼
Pit measurements and QC
```

The thickness map is smoothed using a Gaussian kernel defined in physical units:

```python
FOVEA_SMOOTH_SIGMA_MM = 0.10
```

Because A-scan spacing and B-scan spacing are usually very different, the smoothing width is converted separately for each axis.

Missing segmentation values remain missing during smoothing.

A border is excluded from the search:

```python
FOVEA_BORDER_MM = 0.25
```

to reduce the chance that segmentation errors near the acquisition boundary are selected as the fovea.

The detected location is therefore:

```math
(b_f,a_f)
=
\operatorname*{arg\,min}_{b,a}
\tilde{T}(b,a)
```

where $\tilde{T}$ is the smoothed retinal thickness map.

---

### Foveal measurements

After localisation, the pipeline reports:

```text
B-scan index
A-scan index
raw foveal thickness
smoothed minimum thickness
physical x/y position
offset from scan centre
sampling-limited localisation precision
central-subfield thickness
inner-ring thickness
pit contrast
position on the NIR localizer
```

Pit contrast is defined as:

```math
\text{Pit contrast}
=
\text{mean inner-ring thickness}
-
\text{smoothed foveal minimum}
```

A weak pit generates a warning rather than being silently treated as unquestionably valid.

---

## Why the Fovea Algorithm Is a Baseline

The minimum-thickness approach is simple, interpretable, and useful when ILM and BM segmentations are reliable.

However, the globally thinnest retinal point is not guaranteed to be the anatomical fovea in every dataset.

Potential confounders include:

```text
segmentation failure
scan-edge artefacts
retinal pathology
highly tilted posterior poles
abnormal foveal morphology
high/pathological myopia
```

A future version can therefore extend the detector from:

```text
global smoothed minimum
```

to:

```text
candidate pits
      ↓
pit depth / contrast
      ↓
surrounding retinal thickening
      ↓
3D pit morphology
      ↓
inner-retinal-layer support
      ↓
candidate confidence
      ↓
final fovea
```

Possible additional cues include:

- central retinal thinning;
- surrounding parafoveal thickening;
- smooth ILM depression;
- consistency across neighbouring B-scans;
- radial pit symmetry;
- distance from invalid segmentation;
- RNFL/GCL/IPL morphology;
- NIR vascular and optic-disc anatomy.

This distinction is important: the current implementation provides an interpretable **foveal-pit candidate localisation method**, rather than claiming a universally valid anatomical fovea detector.

---

## ETDRS Retinal Thickness Analysis

An ETDRS grid is centred on the detected fovea.

The standard diameters are:

```text
1 mm
3 mm
6 mm
```

corresponding to radii:

```math
0.5,\ 1.5,\ 3.0\text{ mm}
```

The grid contains nine regions:

```text
Central

Inner superior
Inner inferior
Inner nasal
Inner temporal

Outer superior
Outer inferior
Outer nasal
Outer temporal
```

The nasal and temporal assignments are corrected according to laterality:

```text
OD → nasal on image right
OS → nasal on image left
```

All ETDRS masks are generated from the **physical position of the original A-scans**, rather than simply from array indices.

No missing values are interpolated.

---

## ETDRS Measurements

For every ETDRS region the pipeline calculates:

```text
mean thickness
standard deviation
minimum thickness
maximum thickness
retinal volume
valid-data coverage
fraction of the region contained inside the scan
```

Retinal volume is calculated directly from the original thickness samples:

```math
V
=
\sum_i
\left(
\frac{T_i}{1000}
\right)
A_{\text{pixel}}
```

with:

```math
A_{\text{pixel}}
=
scale_x \times scale_z
```

Therefore:

```math
V
=
\sum_i
\left(
\frac{T_i}{1000}
\right)
scale_x\,scale_z
```

where thickness is converted from µm to mm.

Missing regions are not extrapolated.

Instead, incomplete coverage is reported explicitly.

---

## ETDRS Report Figure

The pipeline generates a two-panel ETDRS report.

#### Left panel

```text
Native NIR localizer
+
registered retinal thickness map
+
detected fovea
+
1 / 3 / 6 mm ETDRS grid
```

#### Right panel

A nine-sector ETDRS diagram displaying:

```text
mean retinal thickness
retinal volume
valid-data coverage
```

Additional summary values include:

```text
foveal thickness
central 1-mm mean
central minimum
central maximum
total volume inside the 6-mm circle
overall coverage
```

---

## OCT Depth Projections

The OCT intensity volume can also be projected along the depth axis.

The pipeline calculates:

```text
mean projection
maximum projection
minimum projection
```

for:

```text
the complete A-scan depth

and, when ILM/BM are available,

the retina-only region between ILM and BM
```

Thus six projections can be generated:

```text
Full depth: mean
Full depth: max
Full depth: min

Retina only: mean
Retina only: max
Retina only: min
```

These projections are derived from structural OCT and should **not** be interpreted as OCTA.

---

## En-face Slab Projections

When both required layer boundaries are available, EyePy-defined slab projections can also be generated.

Only slabs whose boundaries are actually segmented are treated as computable.

The resulting arrays are saved numerically as well as displayed as images.

---

## OCT Acquisition Quality

When the `.vol` file contains per-B-scan acquisition-quality values, the pipeline:

- extracts them;
- saves them to CSV;
- calculates the mean quality;
- plots quality against B-scan index.

This can help identify local acquisition degradation or motion-related problems.

---

## OCT Intensity Distribution

The pipeline plots the OCT intensity histogram using a logarithmic count axis.

Zero-valued voxels are excluded from the histogram because large amounts of zero-valued background can otherwise dominate the distribution.

---

## Outputs

For each `.vol` file, the pipeline creates a structured output directory:

```text
vol_outputs/
│
├── batch_summary.csv
│
└── <volume_name>/
    │
    ├── bscans/
    │   ├── bscan_000.png
    │   ├── bscan_001.png
    │   └── ...
    │
    ├── images/
    │   ├── localizer_nir.png
    │   └── enface_<slab>.png
    │
    ├── plots/
    │   ├── 01_localizer_bscan_positions.png
    │   ├── 02_representative_bscans.png
    │   ├── 03_bscan_grid_*.png
    │   ├── 04_segmentation_overlay.png
    │   ├── 05_layer_depth_maps.png
    │   ├── 06_retinal_thickness_map.png
    │   ├── 07_retinal_thickness_on_nir.png
    │   ├── 08_bscan_quality.png
    │   ├── 09_intensity_histogram.png
    │   ├── 10_depth_projections.png
    │   ├── 11_enface_slabs.png
    │   ├── 12_fovea_bscan.png
    │   └── 13_etdrs_retinal_thickness_report.png
    │
    ├── data/
    │   ├── layers.npz
    │   ├── thickness_um.npy
    │   ├── thickness_smoothed_um.npy
    │   ├── depth_projections.npz
    │   ├── fovea.json
    │   ├── etdrs_metrics.json
    │   ├── physical_extents.json
    │   └── bscan_quality.csv
    │
    ├── metadata.json
    ├── report.txt
    └── summary.json
```

---

## Installation

Python 3.10 or newer is recommended.

```bash
pip install eyepy scipy scikit-image imageio matplotlib numpy
```

The pipeline has primarily been designed around:

```text
eyepy >= 0.21
```

---

## Usage

Set the directory containing the `.vol` files:

```python
INPUT_DIR = Path(
    "/path/to/your/vol/files"
)
```

The output directory is defined as:

```python
OUTPUT_ROOT = INPUT_DIR / "vol_outputs"
```

Then run:

```bash
python spectralis_vol_analyzer.py
```

or from Python:

```python
results = run_batch()
```

---

## Configuration

Several behaviours can be controlled near the beginning of the script.

For example:

```python
RECURSIVE = False
OVERWRITE = True
SHOW_PLOTS = False

SAVE_BSCANS = True
SAVE_VOLUME_NPY = False

FOVEA_SMOOTH_SIGMA_MM = 0.10
FOVEA_BORDER_MM = 0.25
FOVEA_MIN_CONTRAST_UM = 40.0

ETDRS_GRID_RADII_MM = (0.5, 1.5, 3.0)
MIN_ETDRS_COVERAGE = 0.95
```

---

## Important Methodological Notes

### Structural OCT, not OCTA

The depth projections generated from the OCT intensity volume are structural OCT projections.

They should not be interpreted as vascular OCTA maps.

---

### Missing data

Missing segmentation values are represented as `NaN`.

They are not filled when calculating thickness, foveal measurements, ETDRS statistics, or volume.

---

### Physical scale

The `.vol` file supplies device-based lateral and axial scaling.

Current ETDRS distances therefore use the device-provided retinal scale.

They are **not corrected for individual axial length / ocular magnification**.

This distinction may be particularly relevant for highly myopic eyes.

---

### Retinal thickness direction

Current retinal thickness is calculated along the OCT axial/image-column direction:

```math
BM_y-ILM_y
```

This is not necessarily identical to the shortest surface-normal distance between ILM and BM.

For ordinary macular imaging this may be a reasonable approximation, but surface-normal thickness may become relevant for strongly tilted or highly curved posterior poles, particularly in high myopia.

---

## Research Context

This pipeline was developed as part of exploratory OCT analysis for research involving retinal imaging and quantitative macular morphology.

Potential applications include:

```text
retinal thickness analysis
myopia research
longitudinal OCT analysis
macular morphology
retinal biomarker extraction
image-quality assessment
segmentation validation
machine-learning preprocessing
```

The current focus is on transparent and interpretable feature extraction rather than black-box prediction.

---

## Future Development

Planned or possible extensions include:

**More robust fovea localisation**

Move from a single smoothed global minimum toward candidate-based foveal-pit detection incorporating:

```text
pit contrast
pit width
3D curvature
radial symmetry
adjacent-B-scan consistency
inner-retinal-layer thinning
NIR anatomical support
confidence scoring
```

**Additional retinal-layer measurements**

With sufficiently complete segmentation:

```text
RNFL
GCL
IPL
GCIPL
GCC
INL
OPL
ONL
outer retina
photoreceptor / RPE-associated measurements
```

**Choroidal analysis**

If the choroid–sclera interface can be segmented reliably:

```text
subfoveal choroidal thickness
choroidal thickness maps
ETDRS choroidal measurements
```

**Axial-length correction**

When axial length is available, lateral OCT dimensions could be corrected for ocular magnification before ETDRS and spatial analyses.

**Longitudinal analysis**

Repeated visits could be registered and compared using:

```math
\Delta T=T_{t_2}-T_{t_1}
```

and, where axial length is available:

```math
\Delta AL=AL_{t_2}-AL_{t_1}
```

to investigate structural changes over time.

---

## Reproducibility

The pipeline intentionally saves both figures and underlying numerical arrays.

This makes it possible to:

- reproduce plots without re-reading the `.vol` file;
- inspect derived measurements independently;
- audit fovea localisation;
- verify ETDRS values;
- use the numerical outputs for downstream statistics or machine learning.

---

## Limitations

This repository should currently be considered a **research and exploratory analysis tool**, not a clinical diagnostic system.

Results depend on:

```text
OCT acquisition quality
correct file metadata
segmentation accuracy
scan coverage
appropriate macular acquisition
device calibration
```

Automatic fovea localisation should therefore be visually reviewed, particularly in abnormal anatomy, severe myopia, retinal pathology, or incomplete segmentation.

---

## Dependencies

Core Python packages:

```text
numpy
scipy
matplotlib
scikit-image
imageio
eyepy
```

---

## Acknowledgements

This project uses **EyePy**, an open-source Python library for ophthalmic imaging developed by the Medical Image Analysis Group at the University of Bonn.

EyePy provides support for loading and working with Heidelberg ophthalmic imaging formats and forms the core file-reading layer of this pipeline.

---

## Disclaimer

This software is intended for **research use only**.

It is not a medical device and must not be used as a substitute for clinical interpretation, diagnosis, treatment decisions, or validated ophthalmic software.

---

### Citation

If you use this repository in academic work, please cite the repository and the relevant EyePy publications/software release.

A formal citation for this project can be added once the methodology and repository version are finalised.
