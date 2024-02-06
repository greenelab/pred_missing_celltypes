# general imports
import warnings
import numpy as np
import os
import pandas as pd
import sklearn as sk
import scipy as sp
from scipy.sparse import coo_matrix

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter

# file processing
import pickle
import gzip
from pathlib import Path


#function to sort columns alphabetically in a DataFrame
def sort_columns_alphabetically(df):
    return df.reindex(sorted(df.columns), axis=1)


#write files for CIBERSORTX
def write_cibersortx_files(bp_path, out_file_id, sig_df, X_train, num_str, bulks_type):
  #make index compatible w/ R

  sig_df= sig_df.T

  #save
  sc_profile_file = os.path.join(bp_path, f"{out_file_id}_{bulks_type}_{num_str}missing_signal.txt")
  sc_profile_path = Path(sc_profile_file)
  sig_df.to_csv(sc_profile_path, sep='\t', index=True) 
  #and save mixture file
  #transpose incoming X_train file
  pseudos_df = X_train.transpose()
  pseudos_df.columns = range(pseudos_df.shape[1])
  print(sig_df.shape)
  return(pseudos_df, sig_df)

#write files for bayes prism
def write_bp_files(bp_path, out_file_id, sig_df, X_train, num_str, bulks_type):
  #make index compatible w/ R
  sig_df.index = range(1, len(sig_df)+ 1)
  sig_df = sig_df.transpose()
  #save
  sc_profile_file = os.path.join(bp_path, f"{out_file_id}_{bulks_type}_{num_str}missing_signal.csv")
  sc_profile_path = Path(sc_profile_file)
  sig_df.to_csv(sc_profile_path, index= True)
  #and save mixture file
  #make index compatible w/ R
  X_train.index = range(1, len(X_train)+ 1)
  #transpose incoming X_train file
  pseudos_df = X_train.transpose()
  pseudos_df.columns = range(pseudos_df.shape[1])
  
  return(pseudos_df, sig_df)

# select cells to delete with random seed
def select_cells_missing(sn_adata, num_cells_missing, random_seed):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # get the unique cell types in sn_adata
    cell_types = sn_adata.obs['cell_types'].unique()
    
    # initialize the dictionary to store the selected cells for each num_cells_missing
    cells_to_miss = {}
    if 0 in num_cells_missing:
      cells_to_miss[0] = []
    
    # sort num_cells_missing in ascending order
    num_cells_missing.sort()
    
    # initialize an empty set to keep track of selected cells for all num_cells_missing values
    selected_cells = set()
    
    for num in num_cells_missing:
        # ensure num is within the range of cell types
        if num >= len(cell_types):
            raise ValueError("num_cells_missing exceeds the number of available cell types.")
        
        # randomly select num cells that haven't been selected before
        num_to_select = num - len(selected_cells)
        if num_to_select > 0:
            available_cells = list(set(range(len(cell_types))) - selected_cells)
            if num_to_select >= len(available_cells):
                selected = available_cells
            else:
                selected = np.random.choice(available_cells, num_to_select, replace=False)
            selected_cells.update(selected)
            cells_to_miss[num] = list(selected_cells)  # store selected cells as a list
    
    return cells_to_miss

#the following functions are adapted from https://github.com/greenelab/sc_bulk_ood/blob/main/sc_preprocessing/sc_preprocess.py
# cell type specific pseudobulk
def get_cell_type_sum(in_adata, cell_type_id, num_samples):

  # get the expression of the cells of interest
  cell_df = in_adata[in_adata.obs["scpred_CellType"].isin([cell_type_id])]

  # now to the sampling
  cell_sample = sk.utils.resample(cell_df, n_samples = num_samples, replace=True)

  sum_per_gene = cell_sample.X.sum(axis=0)

  return sum_per_gene

def generate_random_vector(len_vector, num_cells):
  # Generate random numbers from a lognormal distribution
  rand_numbers = np.random.lognormal(size=len_vector)
  
  # Make sure all numbers are positive
  rand_numbers = np.abs(rand_numbers)
  
  # Scale the numbers to sum up to num_cells
  rand_vec = (rand_numbers / np.sum(rand_numbers)) * num_cells
  
  # Round the numbers to integers
  rand_vec = np.round(rand_vec).astype(int)
  
  # Ensure there are no exact zeros
  rand_vec[rand_vec == 0] = 1
  
  # Adjust the sum if needed due to rounding
  diff = num_cells - np.sum(rand_vec)
  if diff > 0:
      rand_vec[np.argmax(rand_vec)] += diff
  elif diff < 0:
      rand_vec[np.argmin(rand_vec)] += diff
    
  return rand_vec

# method to generate a proportion vector
def gen_prop_vec_lognormal(len_vector, num_cells):

  rand_vec = np.random.lognormal(5, np.random.uniform(1,3), len_vector) # 1

  rand_vec = np.round((rand_vec/np.sum(rand_vec))*num_cells)
  
  if(np.sum(rand_vec) != num_cells):
    idx_change = np.argmax(rand_vec)
    rand_vec[idx_change] = rand_vec[idx_change] + (num_cells - np.sum(rand_vec))

  rand_vec = rand_vec.astype(int)
  
  return rand_vec

# method to generate true proportion vector
def true_prop_vec(in_adata, num_cells):

  rand_vec = in_adata.obs["scpred_CellType"].value_counts() / in_adata.obs["scpred_CellType"].shape[0]
  rand_vec = np.array(rand_vec)

  rand_vec = np.round(rand_vec*num_cells)
  if(np.sum(rand_vec) != num_cells):
    idx_change = np.argmax(rand_vec)
    rand_vec[idx_change] = rand_vec[idx_change] + (num_cells - np.sum(rand_vec))

  rand_vec = rand_vec.astype(int)
  
  return rand_vec

def calc_prop(in_adata, cell_order):

  tab = in_adata.obs.groupby(['scpred_CellType']).size()
  tab = tab[cell_order]
  real_prop = np.asarray(tab/np.sum(tab))
  prop_cols = np.asarray(cell_order)

  props = pd.DataFrame(real_prop)
  props = props.transpose()
  props.columns = prop_cols      

  return props

def get_single_celltype_prop_matrix(num_samp, cell_order):
  num_celltypes = len(cell_order)  

  total_prop = pd.DataFrame(columns = cell_order)


  for curr_cell_idx in range(num_celltypes):
    curr_prop = [0.01]*num_celltypes
    curr_prop[curr_cell_idx] = 1

    curr_cell_prop_df = get_corr_prop_matrix(num_samp, curr_prop, cell_order, min_corr=0.95)
    total_prop = total_prop.append(curr_cell_prop_df)

  return total_prop

# total pseudobulk
def use_prop_make_sum(in_adata, num_cells, props_vec, cell_noise, sample_noise, noise_type):

  len_vector = props_vec.shape[1]
  cell_order = props_vec.columns.values.to_list()

  # instantiate the expression and proportion vectors
  total_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  total_prop = pd.DataFrame(columns = cell_order)

  # cell specific noise, new noise for each sample
  if cell_noise is None:
    cell_noise = [np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0]) for i in range(len_vector)]

  # iterate over all the samples we would like to make
  for samp_idx in range(props_vec.shape[0]):
    if samp_idx % 100 == 0:
        print(samp_idx)

    n_cells = num_cells

    if num_cells is None:
      n_cells = np.random.uniform(200, 5000)

    props = pd.DataFrame(props_vec.iloc[samp_idx])
    props = props.transpose()
    props.columns = cell_order

    sum_over_cells = np.zeros(in_adata.var['gene_ids'].shape[0])

    #iterate over all the cell types
    for cell_idx in range(len_vector):
      cell_type_id = cell_order[cell_idx]
      num_cell = props_vec.iloc[samp_idx, cell_idx]*n_cells
      num_cell = num_cell.astype(int)
      ct_sum = get_cell_type_sum(in_adata, cell_type_id, num_cell)

      # add noise if we don't want the true proportions
      #if not use_true_prop:
      ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])

      sum_over_cells = sum_over_cells + ct_sum

    sum_over_cells = pd.DataFrame(sum_over_cells)

    sum_over_cells.columns = in_adata.var['gene_ids']
    #sum_over_cells = sum_over_cells.T
    # sample specific noise
    # add sample noise
    if noise_type == "All noise":
      if sample_noise is None:
        sample_noise = np.random.lognormal(0, 1, in_adata.var['gene_ids'].shape[0])  # 0.1
      else: 
        sample_noise = sample_noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
      # library size
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, 1)[0]
      # random variability
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])
      # add poisson noise
      sum_over_cells = np.random.poisson(sum_over_cells)[0]
    elif noise_type == "No noise":
      sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0])  #empty noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
    elif noise_type == "No sample noise":
      sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0]) #empty noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
      # library size
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, 1)[0]
      # random variability
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])
      # add poisson noise
      #sum_over_cells = np.random.poisson(sum_over_cells)[0]
      sum_over_cells = sum_over_cells *np.random.poisson(sum_over_cells)[0]


    sum_over_cells = pd.DataFrame(sum_over_cells)
    if len(sum_over_cells.columns) != len(in_adata.var['gene_ids']):
      sum_over_cells = sum_over_cells.T
    sum_over_cells.columns = in_adata.var['gene_ids']

    total_expr = total_expr.append(sum_over_cells)
    total_prop = total_prop.append(props)

  return (total_prop, total_expr, sample_noise)

# total pseudobulk
def make_prop_and_sum(in_adata, num_samples, num_cells, use_true_prop, cell_noise, sample_noise, noise_type):

  len_vector = in_adata.obs["scpred_CellType"].unique().shape[0]

  # instantiate the expression and proportion vectors
  total_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  total_prop = pd.DataFrame(columns = in_adata.obs["scpred_CellType"].unique())

  test_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  test_prop = pd.DataFrame(columns = in_adata.obs["scpred_CellType"].unique())


  # cell specific noise, new noise for each sample
  if cell_noise is None:
    cell_noise = [np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0]) for i in range(len_vector)]

  # iterate over all the samples we would like to make
  for samp_idx in range(num_samples+100):
    if samp_idx % 100 == 0:
      print(samp_idx)

    n_cells = num_cells
    if num_cells is None:
      n_cells = np.random.uniform(200, 5000)

    if use_true_prop is True:
      props_vec = true_prop_vec(in_adata, n_cells)
    else:
      props_vec = generate_random_vector(len_vector, n_cells)
    props = pd.DataFrame(props_vec)
    props = props.transpose()
    props.columns = in_adata.obs["scpred_CellType"].unique()

    sum_over_cells = np.zeros(in_adata.var['gene_ids'].shape[0])

    #iterate over all the cell types
    for cell_idx in range(len_vector):
      cell_type_id = in_adata.obs["scpred_CellType"].unique()[cell_idx]
      num_cell = props_vec[cell_idx]
      ct_sum = get_cell_type_sum(in_adata, cell_type_id, num_cell)

      # add noise if we don't want the true proportions
      #if not use_true_prop:
      ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])
      sum_over_cells = sum_over_cells + ct_sum

    sum_over_cells = pd.DataFrame(sum_over_cells)
    if len(sum_over_cells.columns) != len(in_adata.var['gene_ids']):
      sum_over_cells = sum_over_cells.T
    sum_over_cells.columns = in_adata.var['gene_ids']  
    
    # add sample noise
    if noise_type == "All noise":
      if sample_noise is None:
        sample_noise = np.random.lognormal(0, 1, in_adata.var['gene_ids'].shape[0])  # 0.1
      else: 
        sample_noise = sample_noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
      # library size
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, 1)[0]
      # random variability
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])
      # add poisson noise
      sum_over_cells = np.random.poisson(sum_over_cells)[0]
    elif noise_type == "No noise":
      sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0])  #empty noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
    elif noise_type == "No sample noise":
      sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0]) #empty noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
      # library size
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, 1)[0]
      # random variability
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])
      # add poisson noise
      #sum_over_cells = np.random.poisson(sum_over_cells)[0]
      sum_over_cells = sum_over_cells *np.random.poisson(sum_over_cells)[0]
      
    sum_over_cells = pd.DataFrame(sum_over_cells)
    if len(sum_over_cells.columns) != len(in_adata.var['gene_ids']):
      sum_over_cells = sum_over_cells.T
    sum_over_cells.columns = in_adata.var['gene_ids']

    if samp_idx < num_samples:
      total_prop = total_prop.append(props)
      total_expr = total_expr.append(sum_over_cells)
    else:
      test_prop = test_prop.append(props)
      test_expr = test_expr.append(sum_over_cells)

  return (total_prop, total_expr, test_prop, test_expr)

# total pseudobulk
def make_prop_and_sum_bulk(in_adata, num_samples, num_cells, use_true_prop, cell_noise, noise_type):
  len_vector = 7

  # instantiate the expression and proportion vectors
  total_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  #total_prop = pd.DataFrame(columns = in_adata.obs["scpred_CellType"].unique())

  test_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  #test_prop = pd.DataFrame(columns = in_adata.obs["scpred_CellType"].unique())

  # cell specific noise, new noise for each sample
  if cell_noise is None:
    cell_noise = [np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0]) for i in range(len_vector)]

  # iterate over all the samples we would like to make
  for samp_idx in range(num_samples+100):
    if samp_idx % 100 == 0:
      print(samp_idx)

    n_cells = num_cells
    if num_cells is None:
      n_cells = np.random.uniform(200, 5000)

    if use_true_prop:
      props_vec = true_prop_vec(in_adata, n_cells)
    else:
      props_vec = generate_random_vector(len_vector, n_cells)
    props = pd.DataFrame(props_vec)
    props = props.transpose()
    #props.columns = in_adata.obs["scpred_CellType"].unique()

    sum_over_cells = np.zeros(in_adata.var['gene_ids'].shape[0])

    #iterate over all the cell types
    for cell_idx in range(len_vector):
      #cell_type_id = in_adata.obs["scpred_CellType"].unique()[cell_idx]
      num_cell = props_vec[cell_idx]
      ct_sum = get_cell_type_sum(in_adata, None, num_cell)

      # add noise if we don't want the true proportions
      if not use_true_prop:
        ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])

      sum_over_cells = sum_over_cells + ct_sum


    sum_over_cells = pd.DataFrame(sum_over_cells)
    sum_over_cells.columns = in_adata.var['gene_ids']

    # add sample noise
    if noise_type == "All noise":
      if sample_noise is None:
        sample_noise = np.random.lognormal(0, 1, in_adata.var['gene_ids'].shape[0])  # 0.1
      else: 
        sample_noise = sample_noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
      # library size
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, 1)[0]
      # random variability
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])
      # add poisson noise
      sum_over_cells = np.random.poisson(sum_over_cells)[0]
    elif noise_type == "No noise":
      sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0])  #empty noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
    elif noise_type == "No sample noise":
      sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0]) #empty noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
      # library size
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, 1)[0]
      # random variability
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])
      # add poisson noise
      #sum_over_cells = np.random.poisson(sum_over_cells)[0]
      sum_over_cells = sum_over_cells *np.random.poisson(sum_over_cells)[0]

    sum_over_cells = pd.DataFrame(sum_over_cells)
    #sum_over_cells = sum_over_cells.T
    sum_over_cells.columns = in_adata.var['gene_ids']

    if samp_idx < num_samples:
      total_prop = total_prop.append(props)
      total_expr = total_expr.append(sum_over_cells)
    else:
      test_prop = test_prop.append(props)
      test_expr = test_expr.append(sum_over_cells)


  return (total_prop, total_expr, test_prop, test_expr)  

def write_cs_bp_files(cibersort_path, out_file_id, sig_df, X_train, patient_idx):
    # write out the scRNA-seq signature matrix
    sig_out_file = os.path.join(cibersort_path, f"{out_file_id}_{patient_idx}_cibersort_sig.tsv.gz")
    sig_out_path = Path(sig_out_file)
    sig_df = sig_df.transpose()

    # cast from matrix to pd
    sig_df = pd.DataFrame(sig_df)

    sig_df.to_csv(sig_out_path, sep='\t',header=False)

    # write out the bulk RNA-seq mixture matrix
    sig_out_file = os.path.join(cibersort_path, f"{out_file_id}_{patient_idx}_cibersort_mix.tsv.gz")
    sig_out_path = Path(sig_out_file)

    X_train.to_csv(sig_out_path, sep='\t',header=True)  
    
    return(X_train, sig_df)

def get_corr_prop_matrix(num_samp, real_prop, cell_order, min_corr=0.8):

  # now generate all the proportions
  total_prop = pd.DataFrame(columns = cell_order)

  while total_prop.shape[0] < num_samp:
    ## generate the proportions matrix
    curr_prop_vec_noise = real_prop*np.random.lognormal(0, 1, len(real_prop))
    curr_prop_vec_noise = np.asarray(curr_prop_vec_noise/np.sum(curr_prop_vec_noise))
    curr_coef = np.corrcoef(curr_prop_vec_noise, real_prop)[0,1]

    if curr_coef > min_corr:
      props = pd.DataFrame(curr_prop_vec_noise)
      props = props.transpose()
      props.columns = cell_order 
      total_prop = total_prop.append(props)

  return total_prop  

def pearsonr_2D(x, y):
    """computes pearson correlation coefficient
       where x is a 1D and y a 2D array"""

    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:,None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:,None], 2), axis=1))
    
    rho = upper / lower
    
    return rho  

def get_prop_matrix_wnoise(orig_df, num_bulks):

    curr_bulk = pd.DataFrame()
    for i in range(0,int(num_bulks)):
        mu, sigma = np.mean(orig_df.iloc[0].values), 0.005 
        # creating a noise with the same dimension as the dataset 
        noise = np.random.normal(mu, sigma, orig_df.shape) 
        signal = (orig_df.values + noise)/2
        curr_bulk = curr_bulk.append(pd.DataFrame(signal, columns = orig_df.columns))   
    return curr_bulk            

#read one file idx
def read_single_pseudobulk_file(data_path, noise_type, file_name, idx):

  pseudobulk_file = os.path.join(data_path, f"{file_name}_{noise_type}pseudo_{idx}.pkl")

  gene_file = os.path.join(data_path, f"{file_name}_intersection_genes.pkl")

  pseudobulk_path = Path(pseudobulk_file)
  gene_path = Path(gene_file)

  pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
  gene_df = pickle.load( open( gene_path, "rb" ) )

  num_samps = pseudobulks_df.shape[0] 

  sample_id = idx
  samp_type = "single"
  prop_file = os.path.join(data_path, f"{file_name}_{noise_type}prop_{idx}.pkl")
  prop_path = Path(prop_file)
  prop_df = pickle.load( open( prop_path, "rb" ) )
  num_cell_type_specific = 50 * prop_df.shape[1]
  cell_prop_type = ["realistic"]*200+["cell_type_specific"]*num_cell_type_specific+['random']*200+['equal']*200

  metadata_df = pd.DataFrame(data = {"sample_id":[sample_id]*num_samps, 
                                    "cell_prop_type":cell_prop_type,
                                    "samp_type":[samp_type]*num_samps,})
  
  return (pseudobulks_df, prop_df, gene_df, metadata_df)

#read all idx files with same name
def read_all_pseudobulk_files(data_path, file_name, num_bulks_training, num_files, noise_type, random_selection):

  X_concat = None
  Y_concat = None
  meta_concat = None

  num = range(0,num_files)
  for idx in num:
    pseudobulks_df, prop_df, gene_df, metadata_df = read_single_pseudobulk_file(data_path, noise_type, file_name, idx)
    print(idx)
    # subsample the number of bulks
    #indexes = metadata_df.groupby("stim").sample(int(num_bulks_training/2))
    #subsamp_idx = indexes.index
    if random_selection:
      subsamp_idx = np.random.choice(range(pseudobulks_df.shape[0]), num_bulks_training)
      pseudobulks_df = pseudobulks_df.iloc[subsamp_idx]
      prop_df = prop_df.iloc[subsamp_idx]
      metadata_df = metadata_df.loc[subsamp_idx]

    X_concat = pd.concat([X_concat, pseudobulks_df])
    Y_concat = pd.concat([Y_concat, prop_df])
    meta_concat = pd.concat([meta_concat, metadata_df])

  return (X_concat, Y_concat, gene_df, meta_concat)