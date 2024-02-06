
'''
EXP2: NNLS, BayesPrism and CIBERSORTx deconvolution of pseudobulks from PBMC3k.
'''


#The following script runs the notebook specified in in_nb with different parameters specified in parameters dict.
#Papermill parametrizes the notebook.
#The resulting notebooks can be found in results_path.

# imports for papermill and path
import papermill as pm
import os

def run_experiment(in_nb, out_nb, parameters):
    """
    Run a notebook using Papermill with specified input, output, and parameters.
    """
    pm.execute_notebook(in_nb, out_nb, kernel_name=ker, parameters=parameters)

def set_common_parameters(data, noise):
    """
    Set common parameters for experiments.
    """
    common_params = {
        "res_name": f"MCT_{data}_EXP2",
        "pseudos_name": f"MCT_{data}_EXP2",
        "path": "/data/",
        "aug_data_path": f"/data/EXP2/{data}/",
        "data_path": "/data/EXP2/",
        "path_results": "/results/EXP2/",
        "num_samples": 10000,
        "noise_type": noise,
        "random_seed": 88,
        "num_missing_cells": [0, 1, 2, 3, 4],
        "nmf_cut": "minimum_value",
        "kernel_name": ker
    }
    return common_params

def run_bayesprism_experiment(data, noise):
    """
    Run BayesPrism experiment with specified data and noise.
    """
    bayesprism_params = set_common_parameters(data, noise)
    bayesprism_params.update({
        "data": "pbmc",
        "bulkprop_type": "random",
        "bp_path": "/data/EXP2/BP_results/"
    })
    run_experiment(
        f'{a_path}/evaluation_experiments/EXP2_bayesprism_eval.ipynb',
        f'{results_path}/EXP2_bayesprism_eval_{data}_{noise}.ipynb',
        bayesprism_params
    )

def run_cibersort_experiment(data, noise):
    """
    Run CIBERSORT experiment with specified data and noise.
    """
    cibersort_params = set_common_parameters(data, noise)
    cibersort_params.update({
        "in_nb": f'{a_path}/evaluation_experiments/EXP2_cibersort_eval.ipynb',
        "data_type": "pbmc3k/",
        "bulkprop_type": "random",
        "cs_path": f"/data/EXP2/cibersort_results/"
    })
    run_experiment(
        cibersort_params["in_nb"],
        f'{results_path}/EXP2_cibersort_eval_{data}_{noise}.ipynb',
        cibersort_params
    )

def run_nnls_experiment(data, noise):
    """
    Run NNLS experiment with specified data and noise.
    """
    nnls_params = set_common_parameters(data, noise)
    nnls_params.update({
        "in_nb": f'{a_path}/evaluation_experiments/EXP2_eval.ipynb',
        "prop_type": 'random',
        "cibersort_files": f"/data/EXP2/cibersort_results/pbmc3k/0_missing/CIBERSORTx_MCT_{data}_EXP2_randomprop_nonoise_0missing_signal_inferred_phenoclasses.CIBERSORTx_MCT_{data}_EXP2_randomprop_nonoise_0missing_signal_inferred_refsample.bm.K999.txt",
        "cells_to_miss_random": True
    })
    run_experiment(
        nnls_params["in_nb"],
        f'{results_path}/EXP2_NNLS_eval_{data}_{noise}.ipynb',
        nnls_params
    )

# Set paths
a_path = os.getcwd()
results_path = f"{a_path}/results/EXP2"
ker = "env_ml"

# Run experiments
run_bayesprism_experiment("pbmc", "nonoise")
run_cibersort_experiment("pbmc", "nonoise")
run_nnls_experiment("pbmc", "nonoise")
