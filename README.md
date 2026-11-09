# African elephant vegetation and drought analyses

Code and prepared inputs for the main figures of **Enhanced vegetation drought resistance associated with elephant disturbance in Africa**.

## Repository structure

| Folder | Contents |
|---|---|
| `config/` | Input requirements, file inventories and analysis settings |
| `data/` | Prepared inputs included with the code; `raw/` is for TIFFs you prepare separately |
| `notebooks/` | Jupyter notebooks for setup and Figs. 1-5; run the workflow here |
| `src/` | Python functions imported by the notebooks |
| `r/` | R script called by notebook 05 when fitting the SEM |
| `results/` | Numerical outputs saved when you run the notebooks |
| `figures/` | PNG and PDF figures exported when you run the notebooks |

The download includes the prepared data and code. The `results/` and `figures/` folders are initially empty and are filled during execution.

## Installation and quick start

Create a Python environment with the required dependencies. The following conda-forge commands provide an example; you may also use an existing environment with these dependencies installed.

```bash
conda create -n african-elephant --override-channels -c conda-forge python=3.9 numpy=1.26.4 pandas=2.3.1 scipy=1.13.1 matplotlib=3.8.4 statsmodels=0.14.5 scikit-learn=1.6.1 seaborn=0.13.2 gdal=3.10.2 basemap=1.4.1 pingouin=0.5.5 pillow jupyterlab ipykernel nbformat nbclient
conda activate african-elephant
python -m ipykernel install --user --name african-elephant
jupyter lab
```

Install **Arial** to reproduce the figure typography.

1. Open [00_setup_and_data.ipynb](notebooks/00_setup_and_data.ipynb), select the kernel associated with your Python environment containing the required dependencies, and run all cells to check the setup.
2. Open a figure notebook below using the same environment. Keep the default settings and select **Restart Kernel and Run All**. Each figure notebook runs independently.
3. Find numerical outputs in `results/` and PNG/PDF figures in `figures/`. Both directories are initially empty.

## Figure notebooks

| Notebook | Output |
|---|---|
| [01_vegetation_structure](notebooks/01_vegetation_structure.ipynb) | Fig. 1 |
| [02_elephant_density](notebooks/02_elephant_density.ipynb) | Fig. 2 |
| [03_temperature](notebooks/03_temperature.ipynb) | Fig. 3 |
| [04_drought_resistance](notebooks/04_drought_resistance.ipynb) | Fig. 4 |
| [05_pathways](notebooks/05_pathways.ipynb) | Fig. 5a-d and a numerical SEM path diagram |

## Input data

**Use the included inputs:** `data/reference/` and `data/covariate_adjustment/` contain prepared inputs derived from the original datasets, including precomputed paired-site results and machine-learning predictions for covariate adjustment. The default workflow uses these inputs without repeating the initial data processing or model training. No TIFF download is needed. See the [data guide](data/README.md) for file contents.

**Start from TIFFs:**

1. Obtain the public source datasets identified in the manuscript and its Data Availability statement. Prepare the 19 inputs listed in [raw_data_require.json](config/raw_data_require.json), following the [raw-data guide](data/raw/README.md), and place them in `data/raw/`. They must share the specified 0.01-degree grid and input scales.
2. In notebook 00, set `CHECK_RAW_GRID=True` and run the input checks.
3. In each of notebooks 01, 03, 04 or 05 that you want to run, set `INPUT_MODE="raw"` and `RUN_ANALYSIS=True`, then select **Restart Kernel and Run All**. Each requires the complete 19-file set. Settings are independent between notebooks.

Raw mode reruns paired-site calculations from the prepared TIFFs. Fig. 1c,d uses the included observations and saved machine-learning predictions without retraining the models; Fig. 2 uses the separate density observations. Prepare the TIFF inputs before running the notebooks, following the raw-data guide above.

## Optional settings

Set input paths and calculation options in each notebook's first code cell; adjust plotting options in its final cell.

| Setting | When to use it |
|---|---|
| `RUN_ANALYSIS=False` | Redraw a figure after its results have already been saved |
| `RUN_BOOTSTRAP=True` (01) | Recalculate confidence intervals instead of using the supplied intervals |
| `COMPUTE_SEM=True` (05) | Fit the SEM again; requires R |
| `DRAW_SEM_DIAGRAM=False` (05, final cell) | Export only Fig. 5a-d |

For a new SEM fit, install R and the packages `lavaan` and `jsonlite` with `install.packages(c("lavaan", "jsonlite"))`. Set `RSCRIPT` in notebook 05 if Rscript is not on your system path. Saved SEM coefficients can be reused only when the input sample matches the supplied reference; otherwise enable `COMPUTE_SEM=True`.

Method settings and validation scope are recorded in [method_notes.md](data/method_notes.md). Citation metadata are in [CITATION.cff](CITATION.cff).
