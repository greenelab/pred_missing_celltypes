# pred_missing_celltypes
### Predicting missing cell-types from deconvolution reference using residual. 

This repository is currently a work in progress of a project aimed at uncovering information from missing cell types, after RNA-bulks have been deconvoluted. Our study aims to answer the question of what happens when we have real and accumulated cells missing from the single-cell dataset in bulk deconvolution methodologies, and whether we are able to retrieve the missing cell-type’s proportions through the calculated proportions. 


## Folder Information:
- /preprocessing/ : notebook to process data and create pseudobulks for experiments.
- /exploratory_experiments/ :  notebooks containing the deconvolution and factorization of different types of pseudobulks.
- /functions/ :  scripts with functions used throughout other folders.
- /data/: contains data information. Specific samples can be downloaded individually as specific in data_information.xlsx.

## Data information:
    - Information on data used can be found in data/data_information.xlsx.

## Note:
- CIBERSORTx was run through Docker with user-specific code using the exploratory_experiments/Run_cibersortx.py script.
- Codes can be given through the CIBERSORTx team by contacting them through the CIBERSORTx website.

The methodology can be roughly divided into experiment 1, 2, 3 and 4. 
The following outline sdescribed each file and the rough outline in which the scripts can be run.

## EXP1: NNLS deconvolution of pseuboulks with distinct immune cell-types.
- /preprocessing/EXP1_pseudos_snadp.ipynb
    - Preprocess and creates pseudobulks from SN adipose tissue data.
- /exploratory_experiments/EXP1.py 
    - Runs EXP1_eval.ipynb through Papermill. Parameters must be specific in EXP1.py file.
- /exploratory_experiments/EXP1_eval.ipynb
    - NNLS deconvolution and residual analysis.    

## EXP2: NNLS, BayesPrism and CIBERSORTx deconvolution of pseudobulks from PBMC3k.
- /preprocessing/EXP2_pseudos_pbmc.ipynb
    - Preprocess and creates pseudobulks from PBMC3k 10x Genomics dataset.
- /preprocessing/EXP2_ciber_bayes_prep.ipynb
    - Creates files needed to run BayesPrism and CIBERSORTx.
- /preprocessing/EXP2_pbmc_BayesPrism.R
    - R script for BayesPrism deconvolution of files created in EXP2_ciber_bayes_prep.ipynb.
- /exploratory_experiments/EXP2.py 
    - Runs EXP2_eval.ipynb through Papermill. Parameters must be specific in EXP2.py file.
- /exploratory_experiments/EXP2_cibersort_eval.ipynb 
    - Analyzes the results of CIBERSORTx deconvolution.
- /exploratory_experiments/EXP2_bayesprism_eval.ipynb 
    - Analyzes the results of BayesPrism deconvolution.
- /exploratory_experiments/EXP2.py 
    - Runs EXP2_eval.ipynb through Papermill. Parameters must be specific in EXP2.py file.
    - This should be run after both BayesPrism and CIBERSORTx are done. 
- /exploratory_experiments/EXP2_eval.ipynb 
    - Final notebook contiaining the NNLS deconvolution and analysis, and the comparison between deconvolution methods. 

## EXP3:
- /preprocessing/EXP3_pseudos_snadp.ipynb
    - Preprocess and creates pseudobulks from white adipose tissue of single-nuclues and single-cell data.
- /preprocessing/EXP3_ciber_bayes_prep.ipynb
    - Creates files needed to run BayesPrism and CIBERSORTx.
- /preprocessing/EXP3_adp_BayesPrism.R
    - R script for BayesPrism deconvolution of files created in EXP3_ciber_bayes_prep.ipynb.
- /exploratory_experiments/EXP3_cibersort_eval.ipynb 
    - Analyzes the results of CIBERSORTx deconvolution.
- /exploratory_experiments/EXP2_bayesprism_eval.ipynb 
    - Analyzes the results of BayesPrism deconvolution.
- /exploratory_experiments/EXP3.py 
    - Runs EXP3_eval.ipynb through Papermill. Parameters must be specific in EXP3.py file.
    - This should be run after both BayesPrism and CIBERSORTx are done. 
- /exploratory_experiments/EXP3_eval.ipynb 
    - Final notebook contiaining the NNLS deconvolution and analysis, and the comparison between deconvolution methods.     
    