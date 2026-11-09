#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: fit_sem.R input.bin input.json fit.rds result.json")
}

if (!requireNamespace("lavaan", quietly = TRUE)) stop("Package 'lavaan' is required")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Package 'jsonlite' is required")

input_bin <- args[[1]]
input_json <- args[[2]]
fit_rds <- args[[3]]
result_json <- args[[4]]
meta <- jsonlite::fromJSON(input_json, simplifyVector = TRUE)

if (!identical(meta$format, "little-endian-float64") || !identical(meta$order, "C-row-major")) {
  stop("Unsupported SEM input binary format")
}
expected_bytes <- as.double(meta$nrow) * as.double(meta$ncol) * 8
actual_bytes <- file.info(input_bin)$size
if (is.na(actual_bytes) || actual_bytes != expected_bytes) {
  stop(sprintf("SEM input size mismatch: expected %.0f, got %s", expected_bytes, actual_bytes))
}
con <- file(input_bin, open = "rb")
values <- readBin(con, what = double(), n = meta$nrow * meta$ncol, size = 8, endian = "little")
close(con)
if (length(values) != meta$nrow * meta$ncol) stop("Could not read all SEM input values")
input <- as.data.frame(matrix(values, nrow = meta$nrow, ncol = meta$ncol, byrow = TRUE))
names(input) <- unlist(meta$columns, use.names = FALSE)

model <- '
tree =~ TH+TC
tree ~ Ele_Dens

ET ~ tree
Albedo ~ tree
LST ~ tree + ET + Albedo
SPEI_m ~ tree

NDVI_res ~ LST + SPEI_m + tree
SIF_res ~ LST + SPEI_m + tree

LST  ~~  SPEI_m
'

seed <- as.integer(meta$seed)
if (is.na(seed)) stop("A fresh-run seed must be recorded in input metadata")
set.seed(seed)
fit <- lavaan::sem(
  model,
  data = scale(input),
  se = "bootstrap",
  bootstrap = as.integer(meta$bootstrap)
)
saveRDS(fit, fit_rds, version = 3)

parameter_estimates <- lavaan::parameterEstimates(fit, standardized = TRUE)
standardized <- lavaan::standardizedSolution(fit, type = "std.all")
keep <- parameter_estimates$op %in% c("=~", "~")
parameter_estimates <- parameter_estimates[keep, , drop = FALSE]
standardized <- standardized[standardized$op %in% c("=~", "~"), , drop = FALSE]
standardized_key <- paste(standardized$lhs, standardized$op, standardized$rhs, sep = "\r")
parameter_key <- paste(parameter_estimates$lhs, parameter_estimates$op, parameter_estimates$rhs, sep = "\r")
match_index <- match(parameter_key, standardized_key)
if (anyNA(match_index)) stop("Could not match standardized SEM paths to parameter estimates")
standardized <- standardized[match_index, , drop = FALSE]

coefficients <- data.frame(
  lhs = parameter_estimates$lhs,
  op = parameter_estimates$op,
  rhs = parameter_estimates$rhs,
  estimate = parameter_estimates$est,
  se = parameter_estimates$se,
  z = parameter_estimates$z,
  pvalue = parameter_estimates$pvalue,
  std_lv = parameter_estimates$std.lv,
  std_all = parameter_estimates$std.all,
  std_nox = parameter_estimates$std.nox,
  std_all_se = standardized$se,
  stringsAsFactors = FALSE
)
fit_measures <- lavaan::fitMeasures(
  fit,
  c("chisq", "df", "pvalue", "cfi", "rmsea", "srmr", "aic")
)

result <- list(
  software = list(R = R.version.string, lavaan = as.character(utils::packageVersion("lavaan"))),
  seed = seed,
  bootstrap = as.integer(meta$bootstrap),
  seed_status = meta$seed_status,
  model = model,
  fit_measures = as.list(fit_measures),
  coefficients = coefficients
)
jsonlite::write_json(
  result,
  result_json,
  auto_unbox = TRUE,
  dataframe = "rows",
  na = "null",
  digits = 17,
  pretty = TRUE
)
