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

def sort_columns_alphabetically(df):
    """
    Sorts the columns of a given pandas DataFrame in alphabetical order.

    Input:
    - df: pandas DataFrame, the input DataFrame whose columns need to be sorted.

    Output:
    - pandas DataFrame with columns sorted alphabetically.
    """
    return df.reindex(sorted(df.columns), axis=1)

def write_cibersortx_files(bp_path, out_file_id, sig_df, X_train, num_str, bulks_type):
    """
    Writes signature and mixture files required for CIBERSORTx.

    Input:
    - bp_path: str, the base path where files will be saved.
    - out_file_id: str, identifier for the output files.
    - sig_df: pandas DataFrame, signature data for CIBERSORTx.
    - X_train: pandas DataFrame, training data to be saved as mixture.
    - num_str: str, a string representing the number of missing cells.
    - bulks_type: str, the type of bulk data used in the files.

    Output:
    - pseudos_df: pandas DataFrame, transposed X_train data.
    - sig_df: pandas DataFrame, transposed signature data (sig_df).
    """
    sig_df = sig_df.T  # Transpose signature data
    
    # Save signature data to file
    sc_profile_file = os.path.join(bp_path, f"{out_file_id}_{bulks_type}_{num_str}missing_signal.txt")
    sc_profile_path = Path(sc_profile_file)
    sig_df.to_csv(sc_profile_path, sep='\t', index=True)

    # Transpose and save training data as mixture file
    pseudos_df = X_train.transpose()
    pseudos_df.columns = range(pseudos_df.shape[1])
    print(sig_df.shape)
    
    return pseudos_df, sig_df

def write_bp_files(bp_path, out_file_id, sig_df, X_train, num_str, bulks_type):
    """
    Writes signature and mixture files required for BayesPrism.

    Input:
    - bp_path: str, the base path where files will be saved.
    - out_file_id: str, identifier for the output files.
    - sig_df: pandas DataFrame, signature data to be used by BayesPrism.
    - X_train: pandas DataFrame, training data to be saved as mixture.
    - num_str: str, a string representing the number of missing cells.
    - bulks_type: str, the type of bulk data used in the files.

    Output:
    - pseudos_df: pandas DataFrame, transposed X_train data.
    - sig_df: pandas DataFrame, transposed signature data (sig_df).
    """
    # Modify the index for compatibility with R and transpose the signature data
    sig_df.index = range(1, len(sig_df) + 1)
    sig_df = sig_df.transpose()

    # Save signature data to file
    sc_profile_file = os.path.join(bp_path, f"{out_file_id}_{bulks_type}_{num_str}missing_signal.csv")
    sc_profile_path = Path(sc_profile_file)
    sig_df.to_csv(sc_profile_path, index=True)

    # Modify index and transpose training data for mixture file
    X_train.index = range(1, len(X_train) + 1)
    pseudos_df = X_train.transpose()
    pseudos_df.columns = range(pseudos_df.shape[1])
    
    return pseudos_df, sig_df

def select_cells_missing(sn_adata, num_cells_missing, random_seed):
    """
    Selects a specified number of unique cell types to remove, based on a random seed.

    Input:
    - sn_adata: AnnData object, containing single-cell data with cell types in the 'obs' attribute.
    - num_cells_missing: list of int, number of cells to remove for each step.
    - random_seed: int or None, the seed for random number generation to ensure reproducibility.

    Output:
    - cells_to_miss: dict, a dictionary where keys are the number of cells missing, and 
      values are lists of indices representing the selected cells to delete.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get the unique cell types from the input data
    cell_types = sn_adata.obs['cell_types'].unique()

    # Dictionary to store selected cells for each number of cells to miss
    cells_to_miss = {}
    if 0 in num_cells_missing:
        cells_to_miss[0] = []

    # Sort the list of cells to delete
    num_cells_missing.sort()

    # Set to keep track of selected cells
    selected_cells = set()

    for num in num_cells_missing:
        # Ensure 'num' is within the valid range of available cell types
        if num >= len(cell_types):
            raise ValueError("num_cells_missing exceeds the number of available cell types.")

        # Calculate how many cells need to be selected for deletion
        num_to_select = num - len(selected_cells)
        if num_to_select > 0:
            available_cells = list(set(range(len(cell_types))) - selected_cells)
            if num_to_select >= len(available_cells):
                selected = available_cells
            else:
                selected = np.random.choice(available_cells, num_to_select, replace=False)
            selected_cells.update(selected)
            cells_to_miss[num] = list(selected_cells)

    return cells_to_miss

def get_cell_type_sum(in_adata, cell_type_id, num_samples):
    """
    Generates the pseudobulk expression by summing gene expression of sampled cells for a specific cell type.

    Input:
    - in_adata: AnnData object, the single-cell data containing gene expression and cell type information.
    - cell_type_id: str or int, the identifier for the cell type of interest to sample.
    - num_samples: int, the number of cells to resample.

    Output:
    - sum_per_gene: numpy array, the summed gene expression values for the sampled cells.
    """
    # Get the expression data for the cells of the specified type
    cell_df = in_adata[in_adata.obs["scpred_CellType"].isin([cell_type_id])]

    # Resample the cells with replacement and sum their gene expression
    cell_sample = sk.utils.resample(cell_df, n_samples=num_samples, replace=True)

    # Sum the gene expression for the sampled cells
    sum_per_gene = cell_sample.X.sum(axis=0)

    return sum_per_gene

def generate_random_vector(len_vector, num_cells):
    """
    Generates a random vector of integer values that sum up to the total number of cells.
    The values are drawn from a lognormal distribution and adjusted to ensure there are no zeros.

    Input:
    - len_vector: int, the length of the random vector (i.e., number of elements).
    - num_cells: int, the total number of cells that the vector should sum to.

    Output:
    - rand_vec: numpy array of integers, the random vector of cell counts that sum to num_cells.
    """
    # Generate random values from a lognormal distribution
    rand_numbers = np.random.lognormal(size=len_vector)
    
    # Ensure all numbers are positive
    rand_numbers = np.abs(rand_numbers)
    
    # Scale the random values to sum up to the specified number of cells
    rand_vec = (rand_numbers / np.sum(rand_numbers)) * num_cells
    
    # Round the values to integers
    rand_vec = np.round(rand_vec).astype(int)
    
    # Ensure there are no exact zeros in the vector
    rand_vec[rand_vec == 0] = 1
    
    # Adjust the sum if rounding caused a difference
    diff = num_cells - np.sum(rand_vec)
    if diff > 0:
        rand_vec[np.argmax(rand_vec)] += diff
    elif diff < 0:
        rand_vec[np.argmin(rand_vec)] += diff
    
    return rand_vec

def gen_prop_vec_lognormal(len_vector, num_cells):
    """
    Generates a proportion vector using a lognormal distribution. The generated values are scaled
    to sum up to the specified number of cells.

    Input:
    - len_vector: int, the length of the proportion vector (number of cell types).
    - num_cells: int, the total number of cells that the vector should sum to.

    Output:
    - rand_vec: numpy array of integers, the proportion vector where each value corresponds
      to the number of cells for a given cell type, and the sum equals num_cells.
    """
    # Generate random values from a lognormal distribution
    rand_vec = np.random.lognormal(5, np.random.uniform(1, 3), len_vector)

    # Scale the values to sum to the total number of cells
    rand_vec = np.round((rand_vec / np.sum(rand_vec)) * num_cells)

    # Adjust the vector if rounding caused a difference in the sum
    if np.sum(rand_vec) != num_cells:
        idx_change = np.argmax(rand_vec)
        rand_vec[idx_change] += (num_cells - np.sum(rand_vec))

    # Convert the values to integers
    rand_vec = rand_vec.astype(int)
  
    return rand_vec

def true_prop_vec(in_adata, num_cells):
    """
    Generates a true proportion vector based on the observed frequencies of cell types in the input data.
    The values are scaled to sum up to the specified number of cells.

    Input:
    - in_adata: AnnData object, containing single-cell data with cell types in the 'obs' attribute.
    - num_cells: int, the total number of cells that the vector should sum to.

    Output:
    - rand_vec: numpy array of integers, the proportion vector where each value corresponds
      to the number of cells for each cell type, and the sum equals num_cells.
    """
    # Get the relative proportions of each cell type from the data
    rand_vec = in_adata.obs["scpred_CellType"].value_counts() / in_adata.obs["scpred_CellType"].shape[0]
    rand_vec = np.array(rand_vec)

    # Scale the proportions to the total number of cells
    rand_vec = np.round(rand_vec * num_cells)

    # Adjust the vector if rounding caused a difference in the sum
    if np.sum(rand_vec) != num_cells:
        idx_change = np.argmax(rand_vec)
        rand_vec[idx_change] += (num_cells - np.sum(rand_vec))

    # Convert the values to integers
    rand_vec = rand_vec.astype(int)
  
    return rand_vec

def get_single_celltype_prop_matrix(num_samp, cell_order):
    """
    Generates a proportion matrix where each cell type is fully represented (proportion 1.0) in each sample,
    while the other cell types are assigned a small proportion (0.01). This matrix is used for creating 
    pseudobulk data with a high correlation for each cell type.

    Input:
    - num_samp: int, the number of samples to generate per cell type.
    - cell_order: list, the list of cell type names, which defines the order of columns in the output matrix.

    Output:
    - total_prop: pandas DataFrame, a matrix where each row represents the proportion of each cell type in a sample.
      For each cell type, there are `num_samp` rows where that cell type has a proportion of 1.0 and all others have 0.01.
    """
    num_celltypes = len(cell_order)  # Number of cell types

    # Initialize the proportion matrix with the specified cell types as columns
    total_prop = pd.DataFrame(columns=cell_order)

    # Generate proportions for each cell type
    for curr_cell_idx in range(num_celltypes):
        # Set the current cell type to 1.0 and others to 0.01
        curr_prop = [0.01] * num_celltypes
        curr_prop[curr_cell_idx] = 1

        # Generate a proportion matrix with high correlation for the current cell type
        curr_cell_prop_df = get_corr_prop_matrix(num_samp, curr_prop, cell_order, min_corr=0.95)

        # Append the generated matrix to the total proportion matrix
        total_prop = total_prop.append(curr_cell_prop_df)

    return total_prop

def use_prop_make_sum(in_adata, num_cells, props_vec, cell_noise, sample_noise, noise_type):
    """
    Generates pseudobulk gene expression data based on input proportions, with optional cell-specific and sample-specific noise.

    Input:
    - in_adata: AnnData object, containing single-cell data with gene expression and cell type information.
    - num_cells: int or None, the total number of cells to simulate per sample. If None, a random value is generated for each sample.
    - props_vec: pandas DataFrame, proportion vector for each sample, specifying the proportions of each cell type.
    - cell_noise: list or None, optional list of noise vectors for each cell type. If None, random noise is generated.
    - sample_noise: numpy array or None, optional noise vector to apply at the sample level. If None, random noise is generated based on noise_type.
    - noise_type: str, the type of noise to apply ("All noise", "No noise", or "No sample noise").

    Output:
    - total_prop: pandas DataFrame, the proportion matrix with rows representing samples and columns representing cell types.
    - total_expr: pandas DataFrame, the gene expression matrix for each sample.
    - sample_noise: numpy array, the noise applied at the sample level.
    """
    len_vector = props_vec.shape[1]  # Number of cell types
    cell_order = props_vec.columns.values.to_list()  # List of cell types

    # Initialize empty DataFrames for storing expression and proportion data
    total_expr = pd.DataFrame(columns=in_adata.var['gene_ids'])
    total_prop = pd.DataFrame(columns=cell_order)

    # Generate cell-specific noise if not provided
    if cell_noise is None:
        cell_noise = [np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0]) for _ in range(len_vector)]

    # Loop through each sample in props_vec
    for samp_idx in range(props_vec.shape[0]):
        if samp_idx % 100 == 0:
            print(samp_idx)

        # Determine the number of cells for the current sample
        n_cells = num_cells if num_cells is not None else np.random.uniform(200, 5000)

        # Extract the proportion vector for the current sample
        props = pd.DataFrame(props_vec.iloc[samp_idx]).transpose()
        props.columns = cell_order

        sum_over_cells = np.zeros(in_adata.var['gene_ids'].shape[0])

        # Loop through each cell type to calculate the expression sums
        for cell_idx in range(len_vector):
            cell_type_id = cell_order[cell_idx]
            num_cell = int(props_vec.iloc[samp_idx, cell_idx] * n_cells)
            ct_sum = get_cell_type_sum(in_adata, cell_type_id, num_cell)

            # Apply cell-specific noise
            ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])

            sum_over_cells += ct_sum

        sum_over_cells = pd.DataFrame(sum_over_cells)
        sum_over_cells.columns = in_adata.var['gene_ids']

        # Apply sample-specific noise based on noise_type
        if noise_type == "All noise":
            if sample_noise is None:
                sample_noise = np.random.lognormal(0, 1, in_adata.var['gene_ids'].shape[0])
            sum_over_cells = np.multiply(sum_over_cells, sample_noise)
            sum_over_cells *= np.random.lognormal(0, 0.1, 1)[0]  # Adjust for library size
            sum_over_cells *= np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])  # Add random variability
            sum_over_cells = np.random.poisson(sum_over_cells)[0]
        elif noise_type == "No noise":
            sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0])  # Empty noise
            sum_over_cells = np.multiply(sum_over_cells, sample_noise)
        elif noise_type == "No sample noise":
            sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0])  # Empty noise
            sum_over_cells = np.multiply(sum_over_cells, sample_noise)
            sum_over_cells *= np.random.lognormal(0, 0.1, 1)[0]  # Adjust for library size
            sum_over_cells *= np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])  # Add random variability
            sum_over_cells *= np.random.poisson(sum_over_cells)[0]

        # Transpose the result if necessary and store it
        if len(sum_over_cells.columns) != len(in_adata.var['gene_ids']):
            sum_over_cells = sum_over_cells.T
        sum_over_cells.columns = in_adata.var['gene_ids']

        total_expr = total_expr.append(sum_over_cells)
        total_prop = total_prop.append(props)

    return total_prop, total_expr, sample_noise

def make_prop_and_sum(in_adata, num_samples, num_cells, use_true_prop, cell_noise, sample_noise, noise_type):
    """
    Generates pseudobulk gene expression data and proportion vectors based on input cell data, 
    with optional noise and varying proportions of cell types.

    Input:
    - in_adata: AnnData object, containing single-cell data with gene expression and cell type information.
    - num_samples: int, the number of samples to generate.
    - num_cells: int or None, the total number of cells per sample. If None, a random value is generated.
    - use_true_prop: bool, whether to use true cell type proportions from the input data or generate random ones.
    - cell_noise: list or None, optional list of noise vectors for each cell type. If None, random noise is generated.
    - sample_noise: numpy array or None, optional noise vector applied at the sample level. If None, random noise is generated.
    - noise_type: str, the type of noise to apply ("All noise", "No noise", or "No sample noise").

    Output:
    - total_prop: pandas DataFrame, the proportion matrix with rows representing samples and columns representing cell types.
    - total_expr: pandas DataFrame, the gene expression matrix for each sample.
    - test_prop: pandas DataFrame, the proportion matrix for test samples (additional samples generated beyond `num_samples`).
    - test_expr: pandas DataFrame, the gene expression matrix for test samples.
    """
    len_vector = in_adata.obs["scpred_CellType"].unique().shape[0]  # Number of unique cell types

    # Initialize empty DataFrames for training and testing expression and proportions
    total_expr = pd.DataFrame(columns=in_adata.var['gene_ids'])
    total_prop = pd.DataFrame(columns=in_adata.obs["scpred_CellType"].unique())
    test_expr = pd.DataFrame(columns=in_adata.var['gene_ids'])
    test_prop = pd.DataFrame(columns=in_adata.obs["scpred_CellType"].unique())

    # Generate cell-specific noise if not provided
    if cell_noise is None:
        cell_noise = [np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0]) for _ in range(len_vector)]

    # Loop through all samples (including 100 extra for testing)
    for samp_idx in range(num_samples + 100):
        if samp_idx % 100 == 0:
            print(samp_idx)

        # Determine the number of cells for the current sample
        n_cells = num_cells if num_cells is not None else np.random.uniform(200, 5000)

        # Generate proportion vector for the current sample
        if use_true_prop:
            props_vec = true_prop_vec(in_adata, n_cells)
        else:
            props_vec = generate_random_vector(len_vector, n_cells)
        props = pd.DataFrame(props_vec).transpose()
        props.columns = in_adata.obs["scpred_CellType"].unique()

        # Sum gene expression over cells
        sum_over_cells = np.zeros(in_adata.var['gene_ids'].shape[0])
        for cell_idx in range(len_vector):
            cell_type_id = in_adata.obs["scpred_CellType"].unique()[cell_idx]
            num_cell = props_vec[cell_idx]
            ct_sum = get_cell_type_sum(in_adata, cell_type_id, num_cell)

            # Apply cell-specific noise
            ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])
            sum_over_cells += ct_sum

        sum_over_cells = pd.DataFrame(sum_over_cells)
        if len(sum_over_cells.columns) != len(in_adata.var['gene_ids']):
            sum_over_cells = sum_over_cells.T
        sum_over_cells.columns = in_adata.var['gene_ids']

        # Apply sample-specific noise based on noise_type
        if noise_type == "All noise":
            if sample_noise is None:
                sample_noise = np.random.lognormal(0, 1, in_adata.var['gene_ids'].shape[0])
            sum_over_cells = np.multiply(sum_over_cells, sample_noise)
            sum_over_cells *= np.random.lognormal(0, 0.1, 1)[0]  # Adjust for library size
            sum_over_cells *= np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])  # Add random variability
            sum_over_cells = np.random.poisson(sum_over_cells)[0]
        elif noise_type == "No noise":
            sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0])  # Empty noise
            sum_over_cells = np.multiply(sum_over_cells, sample_noise)
        elif noise_type == "No sample noise":
            sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0])  # Empty noise
            sum_over_cells = np.multiply(sum_over_cells, sample_noise)
            sum_over_cells *= np.random.lognormal(0, 0.1, 1)[0]  # Adjust for library size
            sum_over_cells *= np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])  # Add random variability
            sum_over_cells *= np.random.poisson(sum_over_cells)[0]

        # Store the generated data into training or testing sets
        if samp_idx < num_samples:
            total_prop = total_prop.append(props)
            total_expr = total_expr.append(sum_over_cells)
        else:
            test_prop = test_prop.append(props)
            test_expr = test_expr.append(sum_over_cells)

    return total_prop, total_expr, test_prop, test_expr

def make_prop_and_sum_bulk(in_adata, num_samples, num_cells, use_true_prop, cell_noise, noise_type):
    """
    Generates pseudobulk gene expression data based on input proportions, with optional cell-specific noise 
    and random noise applied at the sample level. This function focuses on generating bulk data rather than single-cell data.

    Input:
    - in_adata: AnnData object, containing single-cell data with gene expression information.
    - num_samples: int, the number of samples to generate.
    - num_cells: int or None, the total number of cells to simulate per sample. If None, a random value is generated for each sample.
    - use_true_prop: bool, whether to use true cell type proportions from the input data or generate random ones.
    - cell_noise: list or None, optional list of noise vectors for each cell type. If None, random noise is generated.
    - noise_type: str, the type of noise to apply ("All noise", "No noise", or "No sample noise").

    Output:
    - total_prop: pandas DataFrame, the proportion matrix with rows representing samples and columns representing cell types.
    - total_expr: pandas DataFrame, the gene expression matrix for each sample.
    - test_prop: pandas DataFrame, the proportion matrix for test samples (additional samples generated beyond `num_samples`).
    - test_expr: pandas DataFrame, the gene expression matrix for test samples.
    """
    len_vector = 7  # Assuming there are 7 cell types

    # Initialize empty DataFrames for storing expression data for training and testing sets
    total_expr = pd.DataFrame(columns=in_adata.var['gene_ids'])
    test_expr = pd.DataFrame(columns=in_adata.var['gene_ids'])

    # Generate cell-specific noise if not provided
    if cell_noise is None:
        cell_noise = [np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0]) for _ in range(len_vector)]

    # Loop through all samples (including 100 extra for testing)
    for samp_idx in range(num_samples + 100):
        if samp_idx % 100 == 0:
            print(samp_idx)

        # Determine the number of cells for the current sample
        n_cells = num_cells if num_cells is not None else np.random.uniform(200, 5000)

        # Generate proportion vector for the current sample
        if use_true_prop:
            props_vec = true_prop_vec(in_adata, n_cells)
        else:
            props_vec = generate_random_vector(len_vector, n_cells)
        props = pd.DataFrame(props_vec).transpose()

        sum_over_cells = np.zeros(in_adata.var['gene_ids'].shape[0])

        # Loop through each cell type to calculate the expression sums
        for cell_idx in range(len_vector):
            num_cell = props_vec[cell_idx]
            ct_sum = get_cell_type_sum(in_adata, None, num_cell)

            # Apply cell-specific noise if true proportions are not used
            if not use_true_prop:
                ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])

            sum_over_cells += ct_sum

        sum_over_cells = pd.DataFrame(sum_over_cells)
        sum_over_cells.columns = in_adata.var['gene_ids']

        # Apply sample-specific noise based on noise_type
        if noise_type == "All noise":
            if sample_noise is None:
                sample_noise = np.random.lognormal(0, 1, in_adata.var['gene_ids'].shape[0])
            sum_over_cells = np.multiply(sum_over_cells, sample_noise)
            sum_over_cells *= np.random.lognormal(0, 0.1, 1)[0]  # Adjust for library size
            sum_over_cells *= np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])  # Add random variability
            sum_over_cells = np.random.poisson(sum_over_cells)[0]
        elif noise_type == "No noise":
            sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0])  # Empty noise
            sum_over_cells = np.multiply(sum_over_cells, sample_noise)
        elif noise_type == "No sample noise":
            sample_noise = np.random.lognormal(0, 0, in_adata.var['gene_ids'].shape[0])  # Empty noise
            sum_over_cells = np.multiply(sum_over_cells, sample_noise)
            sum_over_cells *= np.random.lognormal(0, 0.1, 1)[0]  # Adjust for library size
            sum_over_cells *= np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])  # Add random variability
            sum_over_cells *= np.random.poisson(sum_over_cells)[0]

        # Store the generated data into training or testing sets
        if samp_idx < num_samples:
            total_prop = total_prop.append(props)
            total_expr = total_expr.append(sum_over_cells)
        else:
            test_prop = test_prop.append(props)
            test_expr = test_expr.append(sum_over_cells)

    return total_prop, total_expr, test_prop, test_expr

def write_cs_bp_files(cibersort_path, out_file_id, sig_df, X_train, patient_idx):
    """
    Writes the scRNA-seq signature matrix and the bulk RNA-seq mixture matrix for use in CIBERSORTx and BayesPrism.

    Input:
    - cibersort_path: str, the base directory where the output files will be saved.
    - out_file_id: str, the identifier for the output files.
    - sig_df: pandas DataFrame, the signature matrix of scRNA-seq data.
    - X_train: pandas DataFrame, the bulk RNA-seq data.
    - patient_idx: str or int, the patient index or identifier.

    Output:
    - X_train: pandas DataFrame, the original bulk RNA-seq data.
    - sig_df: pandas DataFrame, the transposed signature matrix.
    """
    # Write out the scRNA-seq signature matrix
    sig_out_file = os.path.join(cibersort_path, f"{out_file_id}_{patient_idx}_cibersort_sig.tsv.gz")
    sig_out_path = Path(sig_out_file)
    sig_df = sig_df.transpose()  # Transpose signature matrix

    # Save the signature matrix to a file
    sig_df = pd.DataFrame(sig_df)
    sig_df.to_csv(sig_out_path, sep='\t', header=False)

    # Write out the bulk RNA-seq mixture matrix
    sig_out_file = os.path.join(cibersort_path, f"{out_file_id}_{patient_idx}_cibersort_mix.tsv.gz")
    sig_out_path = Path(sig_out_file)
    X_train.to_csv(sig_out_path, sep='\t', header=True)
    
    return X_train, sig_df

def get_corr_prop_matrix(num_samp, real_prop, cell_order, min_corr=0.8):
    """
    Generates a proportion matrix for samples, where each sample's cell type proportions
    are noisy versions of the true proportions, but maintain a specified minimum correlation with the true proportions.

    Input:
    - num_samp: int, the number of samples to generate.
    - real_prop: numpy array, the true proportion vector for cell types.
    - cell_order: list, the list of cell types (column names for the output matrix).
    - min_corr: float, the minimum required Pearson correlation between the generated proportions and the true proportions.

    Output:
    - total_prop: pandas DataFrame, a matrix where each row represents the proportions for a sample,
      and each column corresponds to a cell type.
    """
    # Initialize the total proportion matrix
    total_prop = pd.DataFrame(columns=cell_order)

    # Generate proportions until the number of samples is met
    while total_prop.shape[0] < num_samp:
        # Generate noisy proportions by adding lognormal noise to the real proportions
        curr_prop_vec_noise = real_prop * np.random.lognormal(0, 1, len(real_prop))
        curr_prop_vec_noise = np.asarray(curr_prop_vec_noise / np.sum(curr_prop_vec_noise))  # Normalize

        # Calculate correlation between the noisy and real proportions
        curr_coef = np.corrcoef(curr_prop_vec_noise, real_prop)[0, 1]

        # If the correlation meets the minimum threshold, append the proportions to the matrix
        if curr_coef > min_corr:
            props = pd.DataFrame(curr_prop_vec_noise).transpose()
            props.columns = cell_order
            total_prop = total_prop.append(props)

    return total_prop

def pearsonr_2D(x, y):
    """
    Computes the Pearson correlation coefficient between a 1D array (x) and each row of a 2D array (y).

    Input:
    - x: 1D numpy array, the reference vector for correlation.
    - y: 2D numpy array, where each row will be correlated with the reference vector x.

    Output:
    - rho: 1D numpy array, Pearson correlation coefficients between x and each row of y.
    """
    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:, None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:, None], 2), axis=1))

    rho = upper / lower

    return rho

def get_prop_matrix_wnoise(orig_df, num_bulks):
    """
    Generates a matrix of proportions with added Gaussian noise, simulating noisy bulk data.

    Input:
    - orig_df: pandas DataFrame, the original matrix of proportions (without noise).
    - num_bulks: int, the number of noisy bulk samples to generate.

    Output:
    - curr_bulk: pandas DataFrame, a matrix of noisy bulk proportions with the same columns as the original DataFrame.
    """
    curr_bulk = pd.DataFrame()

    for i in range(int(num_bulks)):
        mu, sigma = np.mean(orig_df.iloc[0].values), 0.005
        # Create Gaussian noise with the same dimensions as the original data
        noise = np.random.normal(mu, sigma, orig_df.shape)
        # Combine original data with noise
        signal = (orig_df.values + noise) / 2
        # Append the noisy bulk data to the result DataFrame
        curr_bulk = curr_bulk.append(pd.DataFrame(signal, columns=orig_df.columns))

    return curr_bulk            

def read_single_pseudobulk_file(data_path, noise_type, file_name, idx):
    """
    Reads a single pseudobulk file and its associated metadata from a specified directory.

    Input:
    - data_path: str, the base directory where the files are located.
    - noise_type: str, the type of noise applied to the pseudobulk data (used in file naming).
    - file_name: str, the base name of the file to read.
    - idx: int, the index of the file to read (used in file naming).

    Output:
    - pseudobulks_df: pandas DataFrame, the pseudobulk data for the specified file index.
    - prop_df: pandas DataFrame, the proportion data for the specified file index.
    - gene_df: pandas DataFrame, the gene information (intersection of genes across samples).
    - metadata_df: pandas DataFrame, metadata describing the pseudobulk sample, including sample ID, cell property type, and sample type.
    """
    # File paths for pseudobulk, genes, and proportions
    pseudobulk_file = os.path.join(data_path, f"{file_name}_{noise_type}pseudo_{idx}.pkl")
    gene_file = os.path.join(data_path, f"{file_name}_intersection_genes.pkl")
    prop_file = os.path.join(data_path, f"{file_name}_{noise_type}prop_{idx}.pkl")

    # Load the pseudobulk, gene, and proportion data from .pkl files
    pseudobulks_df = pickle.load(open(Path(pseudobulk_file), "rb"))
    gene_df = pickle.load(open(Path(gene_file), "rb"))
    prop_df = pickle.load(open(Path(prop_file), "rb"))

    # Number of samples in the pseudobulk data
    num_samps = pseudobulks_df.shape[0]

    # Define metadata for the sample
    sample_id = idx
    samp_type = "single"
    num_cell_type_specific = 50 * prop_df.shape[1]
    cell_prop_type = ["realistic"] * 200 + ["cell_type_specific"] * num_cell_type_specific + ['random'] * 200 + ['equal'] * 200

    # Create a metadata DataFrame
    metadata_df = pd.DataFrame(data={
        "sample_id": [sample_id] * num_samps,
        "cell_prop_type": cell_prop_type,
        "samp_type": [samp_type] * num_samps,
    })

    return pseudobulks_df, prop_df, gene_df, metadata_df

def read_all_pseudobulk_files(data_path, file_name, num_bulks_training, num_files, noise_type, random_selection):
    """
    Reads multiple pseudobulk files with the same base name and concatenates their data into a single dataset. 
    Allows for subsampling of bulk data if specified.

    Input:
    - data_path: str, the base directory where the files are located.
    - file_name: str, the base name of the files to read.
    - num_bulks_training: int, the number of bulks to select per file if random selection is applied.
    - num_files: int, the number of files to read.
    - noise_type: str, the type of noise applied to the pseudobulk data (used in file naming).
    - random_selection: bool, if True, a random selection of bulks will be made from each file.

    Output:
    - X_concat: pandas DataFrame, concatenated pseudobulk data across all files.
    - Y_concat: pandas DataFrame, concatenated proportion data across all files.
    - gene_df: pandas DataFrame, the gene information (intersection of genes across samples).
    - meta_concat: pandas DataFrame, concatenated metadata across all files.
    """
    X_concat = None  # To store concatenated pseudobulk data
    Y_concat = None  # To store concatenated proportion data
    meta_concat = None  # To store concatenated metadata

    # Loop through all file indices
    for idx in range(num_files):
        # Read individual pseudobulk file
        pseudobulks_df, prop_df, gene_df, metadata_df = read_single_pseudobulk_file(data_path, noise_type, file_name, idx)
        print(idx)

        # If random selection is enabled, subsample the bulks
        if random_selection:
            subsamp_idx = np.random.choice(range(pseudobulks_df.shape[0]), num_bulks_training)
            pseudobulks_df = pseudobulks_df.iloc[subsamp_idx]
            prop_df = prop_df.iloc[subsamp_idx]
            metadata_df = metadata_df.loc[subsamp_idx]

        # Concatenate the data across iterations
        X_concat = pd.concat([X_concat, pseudobulks_df])
        Y_concat = pd.concat([Y_concat, prop_df])
        meta_concat = pd.concat([meta_concat, metadata_df])

    return X_concat, Y_concat, gene_df, meta_concat
