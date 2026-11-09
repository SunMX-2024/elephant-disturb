# Method settings and validation

Technical details for interpreting the supplied results. For execution instructions, see the [repository README](../README.md).

### Paired-site inputs

The shared paired-site algorithm is in `src/elephant_analysis/paired_sites.py`; TIFF and binary loading are centralized in `data_inputs.py`. Raw inputs must match the grid specified in `config/raw_data_require.json`. Reference inputs retain the already prepared paired arrays and their original variable names. They allow downstream statistics and figures to be reproduced without rerunning raster preparation or paired-site selection.

Primary paired-site comparisons and samples for the covariate estimators use distinct sampling designs. The internal `nanravel` helper removes NaNs only.

TIFF reads preserve stored pixel values; NoData metadata are not automatically converted. The paired calculation applies source-specific exclusions, including DEM values of -9999, zeros in several inputs, LST values at or below zero, tree cover above 100, and human footprint above 15. Input scales are listed in `config/raw_data_require.json`; the full masking rules remain in [paired_sites.py](../src/elephant_analysis/paired_sites.py).

### Covariate-adjusted contrasts (Fig. 1c,d)

The `afr_251014` archive supplies 15,992 observations in 427 spatial blocks, 27 measured covariates, design weights and aligned out-of-fold propensity/outcome predictions. The NPZ preserves the original 34-column sample matrix, including `block_id`, `row`, `col`, exposure and outcomes. The `yhat_oof` vectors remain as audit fields; the four estimators use propensity scores and `m1`/`m0` predictions. Outcomes are ordered TH, TC; methods are Matching, IPW, Standardization, DR.

The arrays preserve source row order and float64 precision. Conversion settings and source hashes are recorded in `data/covariate_adjustment/covariate_inputs_manifest.json`; no CSV or pickle-based object arrays are required at runtime.

The implementations preserve the source estimator definitions:

- **ATT-Match:** custom within-block matching, propensity clipped to 0.1-0.9, logit caliper 0.1 times the within-block logit standard deviation, and up to five controls with replacement. Each event weight is the treated design weight divided by the control's running reuse count.
- **ATE-IPW:** within-block design-weighted treatment proportions stabilize weights; propensity is clipped to 0.1-0.9 and raw weights are winsorized at unweighted 10th/90th percentiles. Arms are normalized within each block and block contrasts are combined by total design weight.
- **ATE-Gcomp:** design-weighted mean of `m1 - m0` across the full sample.
- **ATE-DR:** augmented residuals are trimmed at the 10th/90th percentiles and combined with `m1 - m0` using the design-weighted Hajek mean. TC uses unweighted trimming; TH uses design-weighted trimming for the point estimate.

The eight recalculated point estimates match the saved reference. Default 95% confidence intervals are saved percentile-bootstrap results (100 draws). The supplied bootstrap keeps original block IDs on repeated draws, so subsequent grouping combines repeated blocks. It uses unweighted DR trimming for both outcomes, whereas the TH point estimate uses weighted trimming. These are the source implementation's settings; the intervals have not been independently revalidated. Refitting nuisance models is outside this workflow.

### Density observations (Fig. 2)

`density_inputs.npz` supplies country summaries and patch-density observations, with their column names, row order and source hashes. The observation compilation is an input to this repository; its upstream construction is not included in the raster workflow.

Country panels use archived summary arrays, ordinary least squares with 95% mean confidence intervals, 5,000-permutation slope tests (seed 42), and the original side distributions and zero-centred Wilcoxon test. Patch panels retain adaptive quantile bins (`min_pts=15`, `min_bins=5`, `max_bins=50`) and the original regression/permutation settings.

### Partial correlations and SEM (Fig. 5)

Samples retain cells with paired DEM differences no greater than 200 m and slope differences no greater than 10 degrees. NDVI and SIF use separate resistance masks and complete cases. Protected cells precede unprotected cells in C-order. SPEI is the pair mean; other partial-correlation panel variables are elephant-affected minus comparison values.

Partial correlations use Pingouin with Spearman ranks. Conditional trend lines use no-intercept OLS residual vectors, so the displayed trends and reported partial-rank statistics describe different calculations. Numerical equivalence to the earlier notebook's local rank/residual/Pearson approximation has not been established.

The SEM matrix has 1,381 rows and nine columns. It uses both resistance masks, complete cases, paired mean SPEI and elephant-side density. The prepared reference-mode matrix matches the archived input exactly. The notebook checks sample values and ordering before reusing archived coefficients. Changed samples require a new fit.

`r/fit_sem.R` supplies the model specification, scaling and 5,000-bootstrap lavaan settings. Python and R exchange a float64 binary matrix and JSON metadata; R saves the fitted model as RDS and coefficients and fit information as JSON. The saved reference identifies lavaan 0.6-15 but does not record the R version or bootstrap seed. New fits use the notebook's specified seed. The reference NPZ contains coefficients, standard errors and p-values; its fit-index arrays are empty.

### Figure rendering

The plotting cells for Figs. 1-4 and Fig. 5a-d retain the source dimensions, fonts, labels, colours and layout. Source filenames and checksums are recorded in [source_hashes.json](../config/source_hashes.json). The Fig. 4 protected-area mask is used only for display at its original 0.01-degree resolution, with zeros transparent.

The Fig. 1c,d axes retain the source label `Effects of elephants`; the adjusted estimates are interpreted as observational associations. Distribution annotations in Figs. 1, 3 and 4 use `kstest(values, "norm")` on the plotted differences, testing agreement with the fixed standard normal distribution N(0,1). These are not paired-contrast tests or tests of normality with fitted mean and variance.

The SEM diagram maps `lhs ~ rhs` to an arrow from predictor `rhs` to response `lhs`, and `lhs =~ rhs` from latent variable to indicator. Each arrow reads its `std.all` coefficient by path identity. Significance comes from the exported parameter-estimate p-value, with thresholds applied before rounding; a fixed unstandardized marker loading is labelled with a dagger rather than a significance star. Only regression paths and measurement loadings are shown: the model includes `LST ~~ SPEI_m`, but the saved directed-path export does not contain that covariance. The complete model remains in `r/fit_sem.R`. The displayed values and full-precision parameters are written to `results/sem_paths.json`.

### Validation scope

All six notebooks passed with default bundled inputs on 2026-11-08 in the existing Windows environment. The eight adjusted point estimates and 1,381 x 9 SEM sample matrix match the saved references. These checks cover input conversion and downstream execution. A fresh environment installation, the complete raw-raster route, nuisance-model training and full bootstrap refits were not tested. The analyses describe observational associations.

Notebook 05 was rerun on 2026-11-09 with the numerical SEM diagram. Its Fig. 5a-d PNG is unchanged, and all 15 displayed path coefficients and parameter-test p-values match the source exports. The diagram checks path identities against the supplied model and treats the fixed marker loading separately from significance tests.

Notebook 01 was also rerun after updating the input folder name to `covariate_adjustment/`; its numerical results and PNG outputs remain identical.

On 2026-11-09, notebook 00 passed with the revised 19-file input catalog and no raw TIFFs present. Synthetic 2 x 3 TIFFs verified manifest-based loading, header-only setup checks, and rejection of missing or incompatible inputs. These checks do not validate a complete continental raw-data run or upstream data preparation.

A full default run on 2026-11-09 generated all main-figure panels and the numerical SEM diagram. All eight numerical result bundles matched the earlier validated outputs exactly. Fig. 3c,d retains the original manuscript scatter rendering for visual consistency; its regression annotations use the fitted p-values. Fig. 5 applies significance thresholds to unrounded p-values. Figure dimensions, fonts, labels and numerical estimates are retained. The final export formats are PNG and PDF only (nine figure groups, 18 files). Validation used the existing Windows runtime (including Basemap 2.0.0), not a fresh installation of the example environment in the README.
