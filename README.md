# Data and code for paper: Enhanced vegetation drought resistance induced by elephant disturbance in Africa

## Overview
In this projects, most of image preprocessing, data analysis, and visualization were conducted using python 3.8 in Jupyter Notebook.

This repository contains:
* __Jupyter Notebook scripts__: Found in the code folder, further organized into:
    * code/MS for main manuscript (MS) figures.
    * code/SI for supplementary information (SI) figures.
* __source data__: source-data used in the scripts is available at https://zenodo.org/records/13901576.
    
Each script is fully documented for data entry, calculations, and plotting.

The pairing of each figure and its corresponding script is shown in the table below:
The pairing of each figure and its corresponding script is shown in the table below:

| Fig    | Code file                                                                                     | Remark                    |
|--------|-----------------------------------------------------------------------------------------------|---------------------------|
| Fig.1  | `py_v4_01_forest-structure_240501.ipynb`  <br> `py_v4_05_elephant-density_240504.ipynb`       |                           |
| Fig.2  | `py_v4_02_Lst-difference_240505.ipynb` <br> `py_v4_03_correlation_LST-trees_240506.ipynb`     |                           |
| Fig.3  | `py_v4_03_drought-resistance_240504-Copy1.ipynb`                                              |                           |
| Fig.4  | `py_pathways_analysis_250312.ipynb` <br> `R_sem_240308_update_250312.R`                       | SEM was analyzed in R     |

Other scripts are used for Supplementary Information.


## System Requirements
### Python Dependencies
This projects mainly depends on the Python scientific stack.\
GDAL, numpy, scipy, Pandas, scikit-learn, netCDF4,matplotlib
