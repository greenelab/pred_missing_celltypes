'''
EXP3: NNLS, BayesPrism and CIBERSORTx deconvolution of pseudobulks 
from single-nucleus adipose tissue, with single-cell RNA-seq missing real cell-types. 
Incorporated noise and realistic proportions.
'''



#The following script runs the notebook specified in in_nb with 
# different parameters specified in parameters dict.
#Papermill parametrizes the notebook.
#The resulting notebooks can be found in results_path.

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
    return {
        "res_name": f"MCT_{data}_EXP3",
        "pseudos_name": f"MCT_{data}_EXP3",
        "files_path": "/data/EXP3/",
        "cibersort_files": f"/data/EXP3/cibersort/CIBERSORTx_Job52_MCT_{data}_EXP3_nonoise_cibersort_sig_inferred_phenoclasses.CIBERSORTx_Job52_MCT_{data}_EXP3_nonoise_cibersort_sig_inferred_refsample.bm.K999.txt",
        "rs": 88,
        "num_samples": 10000,
        "nmf_cut_value": "minimum_value",
        "kernel_name": ker
    }

def run_evaluation_experiment(data, noise, bulkprop_type):
    """
    Run evaluation experiment with specified data, noise, and bulkprop_type.
    """
    params = set_common_parameters(data, noise)
    params.update({
        "in_nb": f'{a_path}/evaluation_experiments/EXP3_eval.ipynb',
        "noise_type": noise,
        "bulkprop_type": bulkprop_type
    })
    run_experiment(
        params["in_nb"],
        f'{results_path}/EXP3_{data}_{bulkprop_type}_{noise}.ipynb',
        params
    )

def run_bayesprism_experiment(data, noise, bulkprop_type):
    """
    Run BayesPrism experiment with specified data, noise, and bulkprop_type.
    """
    params = set_common_parameters(data, noise)
    params.update({
        "in_nb": f'{a_path}/evaluation_experiments/EXP3_BayesPrism_eval.ipynb',
        "noise_type": noise,
        "bulkprop_type": bulkprop_type
    })
    run_experiment(
        params["in_nb"],
        f'{results_path}/EXP3_{data}_{bulkprop_type}_{noise}.ipynb',
        params
    )

def run_cibersort_experiment(data, noise, bulkprop_type):
    """
    Run CIBERSORT experiment with specified data, noise, and bulkprop_type.
    """
    params = set_common_parameters(data, noise)
    params.update({
        "in_nb": f'{a_path}/evaluation_experiments/EXP3_cibersort_eval.ipynb',
        "noise_type": noise,
        "bulkprop_type": bulkprop_type
    })
    run_experiment(
        params["in_nb"],
        f'{results_path}/EXP3_{data}_{bulkprop_type}_{noise}.ipynb',
        params
    )

# Set paths
a_path = os.getcwd()
results_path = f"{a_path}/results/EXP3"
ker = "env_ml"

# Run evaluation experiments
run_evaluation_experiment("adp", "nonoise", "random")
run_evaluation_experiment("adp", "nonoise", "realistic")
run_evaluation_experiment("adp", "noise", "random")
run_evaluation_experiment("adp", "noise", "realistic")

# Run BayesPrism experiments
run_bayesprism_experiment("adp", "nonoise", "random")
run_bayesprism_experiment("adp", "nonoise", "realistic")
run_bayesprism_experiment("adp", "noise", "random")
run_bayesprism_experiment("adp", "noise", "realistic")

# Run CIBERSORT experiments
run_cibersort_experiment("adp", "nonoise", "random")
run_cibersort_experiment("adp", "nonoise", "realistic")
run_cibersort_experiment("adp", "noise", "random")
run_cibersort_experiment("adp", "noise", "realistic")