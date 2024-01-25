#The following script runs the notebook specified in in_nb with 
# different parameters specified in parameters dict.
#Papermill parametrizes the notebook.
#The resulting notebooks can be found in results_path.

#imports for papermill and path
import papermill as pm
import nbformat
from nbconvert import PDFExporter
import os, sys

#defining paths for results
a_path = os.getcwd()
results_path = f"{a_path}/results/EXP3"

#environment name
ker = "env_ml"

########################################### EVALUATION ###############################################
###########################################  parameters ##############################################

#starting with NNLS

#notebook to run
in_nb = f'{a_path}/evaluation_experiments/EXP3_eval.ipynb'

#####################
### set the study ###
#####################
#paths:
data = "adp" #adipose data single cell and single nucleus
res_name = f"MCT_adp_EXP3"
pseudos_name = "MCT_adp_EXP3"
files_path = "/data/EXP3/"
cibersort_files = "/data/EXP3/cibersort/CIBERSORTx_Job52_MCT_adp_EXP3_\
nonoise_cibersort_sig_inferred_phenoclasses.CIBERSORTx_Job52_MCT_adp_EXP3_\
nonoise_cibersort_sig_inferred_refsample.bm.K999.txt"
###### set your random seed, num missing cells, and num_samples for reference for reproducibility
rs = 88 #random seed
num_samples = 10000
#how to handle negative residual distribution for NMF
nmf_cut_value = "minimum_value"
kerne_name = ker

###################################################################################################

#Excecuting notebook with no noise.
#random
noise_type = 'nonoise'
bulkprop_type = "random"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{bulkprop_type}_{noise_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)

#realistic
bulkprop_type = "realistic"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{bulkprop_type}_{noise_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)
####################################################################################################

#Excecuting notebook with noise.
#random
noise_type = 'noise'
bulkprop_type = "random"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{noise_type}_{bulkprop_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)
#realistic
noise_type = 'noise'
bulkprop_type = "realistic"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{noise_type}_{bulkprop_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)

####################################################################################################
####################################################################################################

#BayesPrism now

#notebook to run
in_nb = f'{a_path}/evaluation_experiments/EXP3_BayesPrism_eval.ipynb'

#####################
### set the study ###
#####################
#paths:
data = "adp" #adipose data single cell and single nucleus
res_name = f"MCT_adp_EXP3"
pseudos_name = "MCT_adp_EXP3"
files_path = "/data/EXP3/"
###### set your random seed, num missing cells, and num_samples for reference for reproducibility
rs = 88 #random seed
num_samples = 10000
#how to handle negative residual distribution for NMF
nmf_cut_value = "minimum_value"
kerne_name = ker

###################################################################################################

#Excecuting notebook with no noise.
noise_type = 'nonoise'
bulkprop_type = "random"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{bulkprop_type}_{noise_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)

noise_type = 'nonoise'
bulkprop_type = "realistic"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{bulkprop_type}_{noise_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)


####################################################################################################

#Excecuting notebook with noise.
noise_type = 'noise'
bulkprop_type = "random"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{noise_type}_{bulkprop_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)

#Excecuting notebook with noise.
noise_type = 'noise'
bulkprop_type = "realistic"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{noise_type}_{bulkprop_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)

####################################################################################################
####################################################################################################

####################################################################################################
####################################################################################################

#and last, CIBERSORTx:

#notebook to run
in_nb = f'{a_path}/evaluation_experiments/EXP3_cibersort_eval.ipynb'

#####################
### set the study ###
#####################
#paths:
data = "adp" #adipose data single cell and single nucleus
res_name = f"MCT_adp_EXP3"
pseudos_name = "MCT_adp_EXP3"
files_path = "/data/EXP3/"
###### set your random seed, num missing cells, and num_samples for reference for reproducibility
rs = 88 #random seed
num_samples = 10000
#how to handle negative residual distribution for NMF
nmf_cut_value = "minimum_value"
kerne_name = ker

###################################################################################################

#Excecuting notebook with no noise.
noise_type = 'nonoise'
bulkprop_type = "random"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{bulkprop_type}_{noise_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)

noise_type = 'nonoise'
bulkprop_type = "realistic"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{bulkprop_type}_{noise_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)


####################################################################################################

#Excecuting notebook with noise.
noise_type = 'noise'
bulkprop_type = "random"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{noise_type}_{bulkprop_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)

#Excecuting notebook with noise.
noise_type = 'noise'
bulkprop_type = "realistic"
pm.execute_notebook(
   in_nb,
   f'{results_path}/EXP3_{data}_{noise_type}_{bulkprop_type}.ipynb', kernel_name= ker,
   parameters=dict( res_name = res_name, pseudos_name = pseudos_name, 
   files_path = files_path, noise_type = noise_type, cibersort_files = cibersort_files, 
   random_seed = rs, num_samples = num_samples, nmf_cut = nmf_cut_value, kernel_name = ker)
)

####################################################################################################
####################################################################################################