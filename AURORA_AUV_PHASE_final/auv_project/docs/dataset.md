# Dataset Documentation — Bathymetry Processing (Week 2)

## 1. What Is Bathymetry?

Bathymetry is the measurement of depth (or, equivalently, the underwater
"elevation") of the seafloor, analogous to topography on land. A bathymetric
dataset is typically represented as a 2D grid where each cell holds a depth
value for that location.

## 2. Why Bathymetry Is Required for the AUV

The AUV needs to understand the shape of the seafloor before it can plan a
safe path:

* Depth alone tells the AUV how far it is from the seafloor.
* Depth **variation** (slope, roughness, curvature — computed in Week 3)
  reveals hazards such as steep drop-offs, ridges, or rough rocky patches.
* Without a numerical terrain representation, there is no way to build the
  occupancy grid, cost map, or path planner used in later weeks.

Bathymetry is therefore the foundational input to the entire pipeline:

```text
Bathymetry -> Terrain Features -> Terrain Classification -> Occupancy/Cost Map -> A* -> AUV
```

## 3. Data Representation

Internally, bathymetry is represented as a single 2D NumPy array of shape
`(rows, columns)`, dtype `float64`, where each cell value is a depth in
meters. This is the same representation regardless of whether the data came
from a real survey or from the synthetic generator.

## 4. Depth / Elevation Convention

This is one of the easiest things to get silently wrong, so it is handled
explicitly rather than assumed.

Two conventions exist in real-world bathymetric/elevation data:

| Convention | Meaning |
|---|---|
| `positive_down` | Larger positive value = deeper water. `0` = sea surface. This is the project's **internal standard**. |
| `negative_down` | Elevation-style data where more **negative** = deeper (e.g. many GIS elevation rasters store seafloor elevation as negative numbers relative to sea level). |

The convention of the **input** data is set in `configs/config.yaml`:

```yaml
bathymetry:
  depth_convention: "positive_down"   # or "negative_down"
```

`apply_depth_convention()` in `src/data_processing/preprocess.py` converts
any input into the internal `positive_down` standard:

* `positive_down` input → no change.
* `negative_down` input → sign is flipped (`depth = -elevation`).

All downstream modules (Week 3 onward) can therefore assume `positive_down`
without re-checking the convention.

## 5. Preprocessing Pipeline

Implemented in `src/data_processing/preprocess.py`:

```text
raw grid
   │
   ▼
validate_bathymetry()  ── report only, does not modify data
   │
   ▼
clean_bathymetry()
   │  1. Convert +/-Inf → NaN (infinities are never valid depth values)
   │  2. (optional) flag z-score outliers as NaN too
   │  3. Interpolate all NaN cells
   ▼
apply_depth_convention()  ── convert to internal "positive_down" standard
   │
   ▼
(optional) normalize_depth()  ── produces a SEPARATE normalized copy
```

## 6. Missing Value Handling

Configurable via `bathymetry.interpolation_method`:

| Method | Behavior |
|---|---|
| `nearest` | Fill each missing cell with its nearest known neighbor's value. |
| `linear` | Linear interpolation across known points (falls back to nearest-neighbor for missing cells outside the convex hull of known points, e.g. at grid borders). |
| `median` | Fill missing cells with the global median depth. Simple, robust, but ignores spatial structure. |

**Zero is never used as a fallback for missing data**, because zero is a
physically meaningful depth (sea surface) and using it would silently
corrupt the dataset.

### Outlier Handling

A conservative z-score-based outlier detector is available
(`bathymetry.outlier_detection` in `config.yaml`). By default it only
**reports** the number of outliers found — it does not modify real data
unless `auto_filter: true` is explicitly set. This avoids silently distorting
real survey data based on an assumption that might not hold for every dataset.

## 7. Normalization

Two optional normalization methods are available in `normalize_depth()`:

| Method | Formula | Use case |
|---|---|---|
| `minmax` | `(x - min) / (max - min)` → range `[0, 1]` | Good for visualization or ML models sensitive to feature scale. |
| `standardize` | `(x - mean) / std` → zero mean, unit variance | Good for algorithms assuming normally-distributed features. |

**Important:** normalization always produces a **separate array**. The
raw/cleaned depth grid (in real meters) is never overwritten, since later
weeks (slope, cost map, energy estimates) need real physical units.

## 8. Synthetic Environment Generation

Implemented in `src/data_processing/generate_environment.py`,
`generate_synthetic_bathymetry()` builds a terrain grid with genuine spatial
continuity (not random pixel noise) by layering:

1. A constant base depth.
2. A gentle diagonal gradient.
3. Several Gaussian "hills" (shallow mounds).
4. Several Gaussian "depressions" (deep pits).
5. Several elongated "ridges" (raised linear features).
6. Small-amplitude Gaussian noise for local roughness.
7. Clipping to ensure depth stays non-negative.

**⚠️ This is explicitly SYNTHETIC data.** It:

* Does **not** represent any real surveyed location.
* Does **not** claim real-world positional or depth accuracy.
* Is generated purely to let the rest of the pipeline (feature extraction,
  classification, mapping, planning, simulation) be developed and tested
  without requiring an external dataset.

Every environment object produced by `create_environment()` carries a
`metadata["data_origin_note"]` field that states this explicitly, so no
downstream code or report can accidentally present synthetic terrain as real.

## 9. Spatial Resolution

`environment.resolution` in `config.yaml` defines how many meters one grid
cell represents (default: `1.0` → 1 cell = 1 meter). This is a **simulation
assumption**, not a claim about a real surveyed area's resolution. It exists
so later weeks can compute physically-meaningful quantities:

* Path length in meters (Week 6)
* Real-world slope (Week 3)
* Energy-proxy estimates (Week 6)

If a real dataset is integrated with a different native resolution, update
this value to match that dataset's actual cell size.

## 10. Supporting a Real Dataset

The loader (`src/data_processing/load_bathymetry.py`) supports multiple
formats without assuming any single fixed dataset:

| Format | Loader function | Notes |
|---|---|---|
| `.csv` | `load_csv()` | Plain numeric grid, one row of depths per file line. |
| `.npy` | `load_npy()` | Raw NumPy array. |
| `.npz` | `load_npz()` | Archive; tries common key names (`depth`, `elevation`, `z`, `bathymetry`) then falls back to the first array. |
| `.tif` / `.tiff` | `load_geotiff()` | Requires the optional `rasterio` package; raises a clear error if not installed. |

`load_bathymetry(path)` auto-dispatches based on file extension.

To plug in a real dataset:

1. Place the file under `data/raw/`.
2. Set in `configs/config.yaml`:
   ```yaml
   bathymetry:
     source: "real"
     raw_input_path: "data/raw/<your_file>"
     depth_convention: "positive_down"   # or "negative_down" — check your dataset's docs!
   ```
3. Re-run the pipeline: `python -m src.data_processing.run_pipeline`

When a real dataset is added, this section should be updated with:

* **Dataset name:** _(not yet added)_
* **Source:** _(not yet added)_
* **Format:** _(not yet added)_
* **Spatial resolution:** _(not yet added)_
* **Depth units:** _(not yet added)_
* **Preprocessing applied:** _(not yet added)_
* **Train/test split (if used for classifier training in Week 3):** _(not yet added)_
* **Limitations:** _(not yet added)_
* **License / usage restrictions:** _(not yet added)_

## 11. Limitations

* No real bathymetry dataset is currently integrated — all results at this
  stage come from the synthetic generator.
* The synthetic generator uses simple Gaussian/linear features; it does not
  model real geological processes.
* The outlier detector uses a single global z-score threshold, which may not
  suit datasets with highly non-uniform terrain (e.g. a dataset that
  legitimately contains both a shallow reef and a deep trench in the same
  grid). This is a known simplification appropriate for a B.Tech-scope project.
* GeoTIFF support requires the optional `rasterio` dependency, which is not
  installed by default since no GeoTIFF dataset is currently in use.
