#The following script runs the notebook specified in in_nb with different parameters specified in parameters dict.
#Papermill parametrizes the notebook.
#The resulting notebooks can be found in results_path.

#imports for papermill and path
import papermill as pm
import nbformat
from nbconvert import PDFExporter
import os, sys

#defining paths for results
a_path = os.getcwd()
results_path = f"{a_path}/results/EXP2"

#environment name
ker = "env_ml"

###########################################   NNLS   ###############################################
#notebook to run
in_nb = f'{a_path}/evaluation_experiments/EXP2_eval.ipynb'

#random_seed for cell missing selection
rs = 88
#number of cells for creating references equally
num_s = 10000
#how to handle negative distrib. for NMF (either minimim_value or at_0):
nmf_cut_value = "minimum_value"
#data to use
data = "pbmc"
#proportions in bulks
noise = 'nonoise'
prop_type = 'random'
########################
### setting the study ###
#########################
res_name = f"MCT_{data}_EXP2"
pseudos_name = f"MCT_{data}_EXP1"
files_path = "/data/EXP1/"
data_path = "/data/EXP1/"
num_missing_cells = [0,1,2,3,4]
nmf_cut_value = "minimum_value"
prop_type = 'random'
cibersort_files = "/data/EXP1/cibersort/CIBERSORTx_Job55_MCT_snadp_EXP1_0_cibersort_sig_inferred_phenoclasses.CIBERSORTx_Job55_MCT_snadp_EXP1_0_cibersort_sig_inferred_refsample.bm.K999.txt"
################################################
#Excecuting notebook with no noise in PBMC.
#noise added? (noise or nonoise)
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP2_NNLS_eval_{data}_{noise}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, files_path = files_path, noise_type = noise, 
   cibersort_files = cibersort_files, random_seed = rs, num_missing_cells = num_missing_cells, 
   num_samples = num_s, nmf_cut = nmf_cut_value, kernel_name = ker)
)

###########################################   BAYESPRISM   ###############################################
#notebook to run
in_nb = f'{a_path}/evaluation_experiments/EXP2_bayesprism_eval.ipynb'

#random_seed for cell missing selection
rs = 88
#number of cells for creating references equally
num_s = 10000
#how to handle negative distrib. for NMF (either minimim_value or at_0):
nmf_cut_value = "minimum_value"
#data to use
data = "pbmc"
#proportions in bulks
bulkprop_type = "random"
noise = 'nonoise'

########################
### setting the study ###
#########################
res_name = f"MCT_{data}_EXP2"
pseudos_name = f"MCT_{data}_EXP1"
path = "/data/"
aug_data_path = "/data/EXP2/"
data_path = "/data/EXP1/"
bp_path = "/data/EXP2/BP_results/"
num_missing_cells = [0,1,2,3,4]
nmf_cut_value = "minimum_value"

################################################
#Excecuting notebook with no noise in PBMC.
#noise added? (noise or nonoise)
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP2_bayesprism_eval_{data}_{noise}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, path = path,
   aug_data_path = aug_data_path, data_path = data_path, bp_path = bp_path, noise_type = noise,
   random_seed = rs, num_missing_cells = num_missing_cells, num_samples = num_s, nmf_cut = nmf_cut_value, kernel_name = ker)
)

##########################################################################################################
###########################################   CIBERSORTX   ###############################################

#notebook to run
in_nb = f'{a_path}/evaluation_experiments/EXP2_cibersort_eval.ipynb'
#random_seed for cell missing selection
rs = 88
#number of cells for creating references equally
num_s = 10000
#how to handle negative distrib. for NMF (either minimim_value or at_0):
nmf_cut_value = "minimum_value"
#data to use
data = "pbmc"
#proportions in bulks
bulkprop_type = "random"
noise = "nonoise"
########################
### setting the study ###
#########################
res_name = f"MCT_{data}_EXP2"
pseudos_name = f"MCT_{data}_EXP1"
path = "/data/"
aug_data_path = "/data/EXP2/cibersort/"
data_path = "/data/EXP1/"
cs_path = f"{aug_data_path}cibersort_results/"
num_missing_cells = [0,1,2,3,4]
nmf_cut_value = "minimum_value"

###################################################################################################
#Excecuting notebook with no noise in PBMC3k
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP2_cibersort_eval_{data}_{noise}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, prop_type = bulkprop_type,
   path = path, aug_data_path = aug_data_path, data_path = data_path, cibersort_path = cs_path, noise_type = noise,  
   random_seed = rs, num_missing_cells = num_missing_cells, num_samples = num_s, nmf_cut = nmf_cut_value, kernel_name = ker)
)
