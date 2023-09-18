# pred_missing_celltypes
Predicting missing cell-types from deconvolution reference using residual. 

This repository is currently a work in progress of a project aimed at uncovering information from missing cell types, after RNA-bulks have been deconvoluted. 
Currently, NNLS has been used for deconvolution, and the residual is being analyzed as three different matrices (details in exploratory_experiments). The residual is being factoried/reduced using SVD, PCA, ICA and NMF. 

Folders:
preprocessing/ : notebook to process data and create pseudobulks for experiments.
exploratory_experiments/ :  notebooks containing the deconvolution and factorization of different types of pseudobulks.
functions/ :  scripts with functions used throughout.


