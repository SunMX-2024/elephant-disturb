library(lavaan)
library(semPlot)
library(tidyverse)
data <- read.csv('Data/path_analysis_SEM_reduction_250616.csv',header=TRUE)
data2 = scale(data)
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
fit <- sem(model,data=data2,se="bootstrap", bootstrap=5000)
summary(fit,standardized=TRUE, fit.measures=TRUE,rsquare=TRUE)
fitMeasures(fit,c('chisq','df','pvalue','cfi','rmsea','srmr','AIC'))

