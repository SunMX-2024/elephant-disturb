# Input data

The files below are included in the repository. Keep them in their existing locations when running the notebooks. Generated numerical results are saved separately in `results/`. Rendered PNG/PDF figures are included in `figures/` and can be regenerated with the notebooks.

| File | Contents / use |
|---|---|
| `reference/paired_reference.npz` | Precomputed paired-site outputs for Figs. 1a,b, 3, 4 and 5 |
| `reference/density_inputs.npz` | Country and patch-density observations for Fig. 2 |
| `covariate_adjustment/covariate_inputs.npz` | Observations and saved model predictions for Fig. 1c,d |
| `covariate_adjustment/covariate_inputs_manifest.json` | Variable names and provenance for those observations and predictions |
| `reference/covariate_reference.npz` | Saved adjusted estimates and confidence intervals for Fig. 1c,d |
| `reference/protected_area_display.npz` | Background map for Fig. 4 |
| `reference/sem_reference.npz` | Saved SEM input matrix and coefficients for the numerical path diagram |
| `reference/SEM_existing_fit.rds` | Archived R model object; not required for the default Python workflow |

NPZ is a file format, not a processing stage. Specifically, `paired_reference.npz` contains outputs already calculated from paired sites; the other NPZ files contain the observations, predictions, fitted results or map data described above. File checksums and array shapes are listed in [data_manifest.json](../config/data_manifest.json).

## Run with the included inputs

Follow the [quick start](../README.md#installation-and-quick-start) and keep the default notebook settings. Raw TIFFs and R are not required for this route. Notebook 05 exports Fig. 5a-d and a simple SEM path diagram; set `DRAW_SEM_DIAGRAM=False` in its final cell to omit the diagram.

## Raw TIFF inputs

Large TIFFs are excluded from the GitHub repository. Follow [raw/README.md](raw/README.md) to obtain and prepare the inputs listed in [raw_data_require.json](../config/raw_data_require.json), then use raw mode in notebooks 01, 03, 04 or 05.

Obtain the public source datasets identified in the manuscript and its Data Availability statement. Prepare all 19 study inputs using the methods and input specifications provided. The notebooks do not construct these products from provider downloads. Check the complete input list in notebook 00 before running raw mode; similarly named files are not automatically interchangeable.

Raw mode still requires the included model predictions for Fig. 1c,d and the separate density observations for Fig. 2. For a new SEM fit, follow the [R setup instructions](../README.md#optional-settings).

See [method_notes.md](method_notes.md) for the calculation settings, provenance and validation scope.
