# pred_missing_celltypes
### Predicting missing cell-types from deconvolution reference using residual. 

This repository is currently a work in progress of a project aimed at uncovering information from missing cell types, after RNA-bulks have been deconvoluted. Our study aims to answer the question of what happens when we have real and accumulated cells missing from the single-cell dataset in bulk deconvolution methodologies, and whether we are able to retrieve the missing cell-type’s proportions through the calculated proportions. 

## Folder Information:
- `/preprocessing/`: notebook to process data and create pseudobulks for experiments.
- `/exploratory_experiments/`:  notebooks containing the deconvolution and factorization of different types of pseudobulks.
- `/functions/`:  scripts with functions used throughout other folders.
- `/data/`: contains data information. See Data Information below.

## Data information: 
- Information on all data used can be found in: [`Supplemental_Table_2.xlsx`](https://github.com/ivichadriana/pred_missing_celltypes/blob/main/data/Supplemental_Table_2.xlsx)

## Note:
- CIBERSORTx was run through Docker with user-specific code using the exploratory_experiments/Run_cibersortx.py script.
- Codes can be given through the CIBERSORTx team by contacting them through the [CIBERSORTx website](https://cibersortx.stanford.edu/).

The methodology can be roughly divided into experiment 1, 2, 3 and 4. 
The following outline describes each file and the rough outline in which the scripts can be run.

## EXP1: NNLS deconvolution of pseudobulks with distinct immune cell-types.
- `/preprocessing/EXP1_pseudos_snadp.ipynb`
    - Preprocess and creates pseudobulks from SN adipose tissue data.
- `/exploratory_experiments/EXP1.py`
    - Runs `EXP1_eval.ipynb` through Papermill. Parameters must be specific in `EXP1.py` file.
- `/exploratory_experiments/EXP1_eval.ipynb`
    - NNLS deconvolution and residual analysis.    

## EXP2: NNLS, BayesPrism, and CIBERSORTx deconvolution of pseudobulks from PBMC3k.
- `/preprocessing/EXP2_pseudos_pbmc.ipynb`
    - Preprocess and creates pseudobulks from PBMC3k 10x Genomics dataset.
- `/preprocessing/EXP2_ciber_bayes_prep.ipynb`
    - Creates files needed to run BayesPrism and CIBERSORTx.
- `/preprocessing/EXP2_pbmc_BayesPrism.R`
    - R script for BayesPrism deconvolution of files created in `EXP2_ciber_bayes_prep.ipynb`.
- `/exploratory_experiments/EXP2.py`
    - Runs `EXP2_eval.ipynb` through Papermill. Parameters must be specific in `EXP2.py` file.
- `/exploratory_experiments/EXP2_cibersort_eval.ipynb`
    - Analyzes the results of CIBERSORTx deconvolution.
- `/exploratory_experiments/EXP2_bayesprism_eval.ipynb`
    - Analyzes the results of BayesPrism deconvolution.
- `/exploratory_experiments/EXP2.py`
    - Runs `EXP2_eval.ipynb` through Papermill. Parameters must be specific in `EXP2.py` file.
    - This should be run after both BayesPrism and CIBERSORTx are done. 
- `/exploratory_experiments/EXP2_eval.ipynb`
    - Final notebook containing the NNLS deconvolution and analysis, and the comparison between deconvolution methods. 

## EXP3: NNLS, BayesPrism, and CIBERSORTx deconvolution of pseudobulks from single-nucleus adipose tissue, with single-cell RNA-seq missing real cell-types. Incorporated noise and realistic proportions.
- `/preprocessing/EXP3_pseudos_snadp.ipynb`
    - Preprocess and creates pseudobulks from white adipose tissue of single-nucleus and single-cell data.
- `/preprocessing/EXP3_ciber_bayes_prep.ipynb`
    - Creates files needed to run BayesPrism and CIBERSORTx.
- `/preprocessing/EXP3_adp_BayesPrism.R`
    - R script for BayesPrism deconvolution of files created in `EXP3_ciber_bayes_prep.ipynb`.
- `/exploratory_experiments/EXP3_cibersort_eval.ipynb`
    - Analyzes the results of CIBERSORTx deconvolution.
- `/exploratory_experiments/EXP2_bayesprism_eval.ipynb`
    - Analyzes the results of BayesPrism deconvolution.
- `/exploratory_experiments/EXP3.py`
    - Runs `EXP3_eval.ipynb` through Papermill. Parameters must be specific in `EXP3.py` file.
    - This should be run after both BayesPrism and CIBERSORTx are done. 
- `/exploratory_experiments/EXP3_eval.ipynb`
    - Final notebook containing the NNLS deconvolution and analysis, and the comparison between deconvolution methods.     

## EXP4: Real bulks of HGSOC deconvolved with NNLS (See Discussion)
- `/preprocessing/EXP4_preprocessing.ipynb`
    - Preprocess and QCs single-cell  and bulk data from HGSOC.
- `/exploratory_experiments/EXP4_eval.ipynb`
    - NNLS deconvolution and analysis of dissociated and classic bulks.
   
## EXP5: Real bulks of HGSOC deconvolved with NNLS, with Added Adipocytes
- `/preprocessing/EXP4_preprocessing.ipynb`
    - Preprocess and QCs single-cell  and bulk data from HGSOC.
- `/exploratory_experiments/EXP5_eval.ipynb`
    - NNLS deconvolution and analysis of dissociated and classic bulks, with added cells to reference.

## EXP6: Real bulks of Adipose Tissue, deconvolved with NNLS. Uses both scRNA-seq and snRNA-seq.
- `/preprocessing/EXP3_pseudos_snadp.ipynb`
    - Preprocess and creates pseudobulks from white adipose tissue of single-nucleus and single-cell data.
- `/exploratory_experiments/EXP6_eval.ipynb`
    - NNLS deconvolution and analysis of real bulks, with single cell and single nucleus RNA seq as reference.

## Setting up the Conda Environment

To run the experiments in this repository, you'll need to set up a Conda environment named `env_ml`. The required configuration file, `env_ml.yml`, is located in the "environment" folder of this repository.

### Prerequisites

Make sure you have Conda installed. If not, you can download and install it from [Conda's official website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Creating the Conda Environment

Navigate to the root of the repository and run the following command to create the `env_ml` environment:

```bash
conda env create -f environment/env_ml.yml
