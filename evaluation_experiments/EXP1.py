
''' 
EXP1: NNLS deconvolution of pseuboulks with distinct immune cell-types.
'''


#The following script runs the notebook specified in in_nb with different parameters specified in parameters dict.
#Papermill parametrizes the notebook.
#The resulting notebooks can be found in /../results/EXP1/*.
#This file is intended to be run in the folder it resides in. 

#imports for papermill and path
import papermill as pm
import os, sys

#defining paths
a_path = os.getcwd()
results_path = f"{a_path}/results/EXP1"

#environment name
ker = "env_ml"
#notebook to run
in_nb = f'{a_path}/evaluation_experiments/EXP1_eval.ipynb'
#random_seed for cell missing selection
rs = 88
#number of cells for creating references
num_s = 10000
#how to handle negative distrib. for NMF (either minimim_value or at_0):
nmf_cut_value = "minimum_value"

####################################################################################################
# #Excecuting notebook with 5 cell types and all noise.
data = "snadp"
bulkprop_type = ""
num_missing_cells = [0,1,2,3]
csx_file = "MCT_snadp_EXP1_0_cibersort_sig.tsv"

noise_type = "5CTnonoise"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP1_eval_{data}_{noise_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = f"MCT_{data}_EXP1", pseudo_name = f"MCT_{data}_EXP1_{noise_type}", 
   files_path = "/data/EXP1/", noise_type =noise_type,
   cibersort_files = f"/data/EXP1/cibersort_results/{csx_file}",
   random_seed = rs, num_missing_cells = num_missing_cells, num_samples = num_s, nmf_cut = nmf_cut_value, 
   kernel_name = ker)
)

###################################################################################################