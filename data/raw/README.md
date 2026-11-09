# Prepare raw raster inputs

Large raster files are excluded from this GitHub repository because of storage limits. Obtain the public source datasets identified in the manuscript and its Data Availability statement.

Prepare the study products described in the paper and save them in this directory using the filenames in [raw_data_require.json](../../config/raw_data_require.json). Downloading provider files and changing their resolution alone is insufficient: time periods, exposure masks and derived resistance products must also match. These notebooks do not perform that upstream preparation.

## Required grid and values

Supply all 19 single-band TIFFs with the same geographic coordinate reference system and grid:

| Property | Required value |
|---|---|
| Rows x columns | 6,500 x 7,900 |
| Pixel size | 0.01 degrees |
| Upper-left corner | Longitude -26, latitude 25 |
| Extent | Longitude -26 to 53; latitude -40 to 25 |
| GDAL GeoTransform | `[-26, 0.01, 0, 25, 0, -0.01]` |

Match pixel alignment as well as resolution; the notebooks do not resample or reproject inputs.

Preserve the input scales and missing-value conventions specified in the requirements list:

- Supply LST in stored MODIS values, before applying the factor 0.02 and Celsius conversion.
- Supply albedo in stored values, before applying the factor 0.001.
- Use percent for tree cover, metres for tree height and elevation, and degrees for slope. Preserve exposure and protected-area mask codes.
- Preserve the study inputs' missing-value codes. Setting a TIFF's NoData metadata alone does not convert its stored values; see [method_notes.md](../method_notes.md#paired-site-inputs) and the requirements list for details.

## Run the notebooks

1. Place all 19 prepared TIFFs in `data/raw/`. Keep `DATA_ROOT = ROOT / "data"` (or the notebook's existing `REPO_ROOT` equivalent).
2. Open notebook 00 with the kernel for your configured Python environment. Set `CHECK_RAW_GRID=True` and run all cells. Resolve any missing-file or grid errors before continuing; this check does not validate units or study-product definitions.
3. Open notebook 01, 03, 04 or 05 with the same environment. In its first code cell, set:

   ```python
   INPUT_MODE = "raw"
   RUN_ANALYSIS = True
   ```

4. Select **Restart Kernel and Run All**. Each notebook requires the complete 19-file set and can run independently. Changing notebook 00 does not change the settings in figure notebooks.
5. Find numerical outputs in `results/` and plots in `figures/`. After a successful run, set `RUN_ANALYSIS=False` to redraw from saved results.

Fig. 1c,d continues to use the included model predictions. Keep notebook 02 in its default mode to use the included density observations.

For a new SEM fit in notebook 05, also set `COMPUTE_SEM=True` and follow the [R setup instructions](../../README.md#optional-settings). Saved coefficients can be reused only when the prepared SEM sample matches the supplied reference.
