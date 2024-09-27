# general imports
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr, ttest_ind, wilcoxon
from scipy.spatial.distance import euclidean
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from PIL import Image
from collections import Counter
from tqdm import tnrange, tqdm_notebook
import ipywidgets
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scipy as sp
from scipy.optimize import nnls
from scipy import stats
from scipy.stats import spearmanr, pearsonr
# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, MinMaxScaler
from matplotlib_venn import venn2
from upsetplot import from_contents, UpSet
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import MaxNLocator
# programming stuff
import time
import os
import pickle
from pathlib import Path

def capitalize_first_letters(strings):
    """
    Capitalizes the first letter of each string in a list of strings.

    Args:
        strings (list of str): List of strings to be capitalized.

    Returns:
        list of str: A list of strings where the first letter of each string is capitalized.
    """
    return [s.capitalize() for s in strings]

def rmse(y, y_pred):
    """
    Computes the Root Mean Square Error (RMSE) between actual and predicted values.

    RMSE is a standard measure of the difference between values predicted by a model
    and the values actually observed. It is the square root of the average squared differences
    between the actual and predicted values.

    Args:
        y (array-like): Actual observed values.
        y_pred (array-like): Predicted values from the model.

    Returns:
        float: The computed RMSE value.
    """
    #Ensure both y and y_pred are 2D arrays with the same shape
    y = np.array(y).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    
    # Calculate RMSE
    return np.sqrt(((y - y_pred)**2).mean())

def factors_vs_proportions_rmse(factors, 
                                proportions, 
                                num_missing_cells, 
                                method, 
                                plot_r: bool = True):
    """
    Visualizes the relationship between factors and proportions with RMSE and Pearson's correlation analysis.
    
    This function compares the missing cell proportions in bulk RNA-seq data with the factors derived 
    from decomposition methods like PCA, SVD, ICA, or NMF. It generates scatter plots between the 
    cell type proportions and factors, annotating each plot with RMSE and Pearson's correlation coefficient.

    Args:
        factors (dict): A dictionary of factors (e.g., from PCA or NMF) for each missing cell type.
        proportions (dict): A dictionary of proportions of each cell type.
        num_missing_cells (list): A list of missing cell counts to iterate through.
        method (str): The decomposition method used ('PCA', 'SVD', 'ICA', or 'NMF').
        plot_r (bool, optional): Whether to plot the Pearson correlation coefficient 'r'. Default is True.

    Returns:
        None: Generates plots visualizing RMSE and Pearson's correlation between factors and proportions.
    """

    if method == "PCA":
        fc = "PC"
    if method == "SVD":
        fc = "SVD"
    if method == "ICA":
        fc = "IC" 
    if method == "NMF":
        fc = "Factor" 

    # Set the font to Arial for all text
    plt.rcParams['font.family'] = 'Arial'
    # Define your custom colormap colors and their positions
    colors = ['purple', 'white', 'yellow']
    positions = [0.0, 0.5, 1.0]
    # Create the colormap using LinearSegmentedColormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

    # Create a colormap and normalize object for correlation values
    cmap.set_bad((0.4, 0.4, 0.4, 0.4))  # Set alpha value to 0.4 (0 is fully transparent, 1 is fully opaque)
    norm = Normalize(vmin=-1, vmax=1)  # Set vmin and vmax to -1 and 1 for correlations
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])

    # Iterate over the number of missing cells
    for num in num_missing_cells[1:]:
        # Define the number of rows and columns for the grid layout
        num_rows = len(proportions[num].columns)
        num_cols = len(factors[num].columns)

        if num_rows == 1:
            # Create a single subplot with two separate scatter plots
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # Two columns for two factors
            fig.suptitle(f'{method} on Residual: {num} Missing Cell {num} vs. Missing Cell Proportion', fontsize=22, y=0.95)
            x = list(proportions[num].iloc[:, 0])  # Assuming there's only one cell type
            correlations = np.zeros(2)  # Array to store correlations
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1)
            for j, factor in enumerate(factors[num].columns):
                y = list(factors[num][factor])
                ax = axes[j]  # Use the current subplot for plotting

                # Calculate Pearson's correlation coefficient
                r, p = stats.pearsonr(x, y)
                correlations[j] = r
                # Calculate RMSE
                rmse_value = rmse(x, y)
                # Map correlation value to color
                color = scalar_map.to_rgba(r)

                # Scatter plot with color based on correlation
                ax.scatter(x, y, c='dimgrey', alpha=0.7)
                ax.set_xlabel(f'{proportions[num].columns[0]} Proportions', fontsize=16)
                ax.set_ylabel(f'{fc} {factor}', fontsize=16, labelpad = 0.5)
                ax.patch.set_facecolor(color)
                ax.patch.set_alpha(1)
                ax.annotate('RMSE = {:.2f}'.format(rmse_value), xy=(0.5, 0.9), xycoords='axes fraction',
                            ha='center', va='center', fontsize=16, fontweight='bold')
                if plot_r:
                    ax.annotate('r = {:.2f}'.format(r), xy=(0.5, 0.85), xycoords='axes fraction',
                                ha='center', va='center', fontsize=16, fontweight='bold')

                ax.tick_params(axis='both', which='major', labelsize=12)
                
                # Use MaxNLocator to set a maximum number of ticks
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Set max 5 ticks on x-axis
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Set max 5 ticks on y-axis

            # Create a colorbar for the scatter plot
            cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=0.01)
            cax.set_label("Pearson's Correlation (r)", fontsize=18)
            cax.ax.tick_params(size=3, labelsize=12)
            cax.set_alpha(0.4)
        else:
            # Create a grid of subplots
            len_row = 6 * num - num
            len_col = 4 * num - num
            if num == 2:
                len_row = len_row + 2
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(len_row + 10, len_col + 6))
            fig.suptitle(f'{method} on Residual: {num} Missing Cells {num} vs. Missing Cells Proportion', 
                         fontsize=22, y=0.93)  # Adjust the title spacing
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1)
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            # Initialize an array to store correlations
            correlations = np.zeros((num_rows, num_cols))

            # Iterate over cell types and factors
            for i, cell_type in enumerate(proportions[num].columns):
                for j, factor in enumerate(factors[num].columns):
                    x = list(proportions[num][cell_type])
                    y = list(factors[num][factor])
                    # Use the current subplot for plotting
                    ax = axes[i, j]

                    # Calculate Pearson's correlation coefficient
                    r, p = stats.pearsonr(x, y)
                    correlations[i, j] = r

                    # Map correlation value to color
                    color = scalar_map.to_rgba(r)
                    # Calculate RMSE
                    rmse_value = rmse(x, y)
                    # Scatter plot with color based on correlation
                    ax.scatter(x, y, c='dimgrey', alpha=0.7)
                    if len(cell_type) < 8:
                        ax.set_xlabel(f'{cell_type} Proportions', fontsize=16)
                    else:
                        # Adding a new line
                        ax.set_xlabel(f'{cell_type}\nProportions', fontsize=16)
                    ax.set_ylabel(f'{fc} {factor}', fontsize=16, labelpad=0.5)
                    ax.patch.set_facecolor(color)
                    ax.patch.set_alpha(1)
                    ax.annotate('RMSE = {:.2f}'.format(rmse_value), xy=(0.5, 0.9), xycoords='axes fraction',
                                ha='center', va='center', fontsize=12, fontweight="bold")
                    if plot_r:
                        ax.annotate('r = {:.2f}'.format(r), xy=(0.5, .85), xycoords='axes fraction',
                                    ha='center', va='center', fontsize=12, fontweight='bold')

                    ax.tick_params(axis='both', which='major', labelsize=12)

                    # Use MaxNLocator to set a maximum number of ticks
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Set max 5 ticks on x-axis
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Set max 5 ticks on y-axis

            # Create a colorbar for the scatter plots
            if num > 3:
                bar_pad = 0.02
            else:
                bar_pad = 0.01
            cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=bar_pad)
            cax.set_label("Pearson's Correlation (r)", fontsize=18)
            cax.ax.tick_params(size=3, labelsize=12)
            cax.set_alpha(1)

def factors_vs_proportions_heatmaps_real(factors, proportions, num, method, rmse_plot):
    """
    Compares factors to missing cell type proportions and generates heatmaps for real data.

    This function visualizes the correlation between factors (from PCA, SVD, ICA, or NMF) 
    and missing cell type proportions in real bulk RNA-seq data. It produces scatter plots 
    or heatmaps to display the relationship between factors and cell proportions, with 
    annotations for Pearson's correlation coefficient (r) and optionally, Root Mean Square 
    Error (RMSE) if `rmse_plot` is set to True.

    Args:
        factors (dict): A dictionary of factors (e.g., from PCA or NMF) for each missing cell type.
        proportions (dict): A dictionary of proportions of each cell type.
        num (int): Number of missing cells.
        method (str): The decomposition method used ('PCA', 'SVD', 'ICA', or 'NMF').
        rmse_plot (bool): If True, includes RMSE values in the plots. Default is False.

    Returns:
        None: Generates plots comparing factors to cell type proportions.
    """

    if method == "PCA":
        fc = "PC"
    if method == "SVD":
        fc = "SVD"
    if method == "ICA":
        fc = "IC" 
    if method == "NMF":
        fc = "Factor"      
    
    # Create a colormap and normalize object for correlation values
    # Set the font to Arial for all text
    plt.rcParams['font.family'] = 'Arial'
    # Define your custom colormap colors and their positions
    colors = ['purple', 'white', 'yellow']
    positions = [0.0, 0.5, 1.0]
    # Create the colormap using LinearSegmentedColormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))
    # Create a colormap and normalize object for correlation values
    #cmap = plt.get_cmap('cividis')
    cmap.set_bad((0.4, 0.4, 0.4, 0.4))  # Set alpha value to 0.4 (0 is fully transparent, 1 is fully opaque)
    norm = Normalize(vmin=-1, vmax=1)  # Set vmin and vmax to -1 and 1 for correlations
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    # Iterate over the number of missing cells
    
    # Define the number of rows and columns for the grid layout
    num_rows = len(proportions[num].columns) 
    num_cols = len(factors[num].columns)
    
    if num_rows == 1:
        # Create a single subplot with two separate scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(22, 15))  # Two columns for two factors
        fig.suptitle(f'{method} on Residual: {num} Missing Cell {num} vs. Missing Cell Proportion', fontsize=22 ,y=0.95)
        x = list(proportions[num].iloc[:, 0])  # Assuming there's only one cell type
        correlations = np.zeros(2)  # Array to store correlations
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for j, factor in enumerate(factors[num].columns):
            y = list(factors[num][factor])
            ax = axes[j]  # Use the current subplot for plotting

            # Calculate Pearson's correlation coefficient
            r, p = stats.pearsonr(x, y)
            correlations[j] = r
            # Map correlation value to color
            color = scalar_map.to_rgba(r)
            
            # Scatter plot with color based on correlation
            ax.scatter(x, y, c='dimgrey', alpha=0.7)
            if len(cell_type) < 8:
                ax.set_xlabel(f'{cell_type} Proportions',fontsize=22)
            else:
                formatted_label = '\n'.join(cell_type.split())   
                ax.set_xlabel(f'{formatted_label} Proportions',fontsize=22)
            ax.set_ylabel(f'{fc} {factor}', fontsize=22, labelpad = 0.6)
            ax.set_xlabel(f'{proportions[num].columns[0]} Proportions', fontsize=22, labelpad = 0.6)
            ax.patch.set_facecolor(color)
            ax.patch.set_alpha(1)
            ax.tick_params(axis='both', which='major', labelsize=12)
            #only show RMSE if relevant:
            if rmse_plot:
                # Calculate RMSE
                rmse_value = vp.rmse(x, y)
                ax.annotate('RMSE = {:.2f}'.format(rmse_value),xy=(0.5, 0.9), xycoords='axes fraction',
                        ha='center', va='center', fontsize=18, fontweight = 'bold')
            ax.annotate('r = {:.2f}'.format(r), xy=(0.5, .82), xycoords='axes fraction',
                                    ha='center', va='center', fontsize=18, fontweight='bold')
        # Create a colorbar for the scatter plot
        cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=0.01)
        cax.set_label("Pearson's Correlation (r)", fontsize=22)
        cax.ax.tick_params(size=3, labelsize=12)
        cax.set_alpha(0.4)
    else:
        # Create a grid of subplots
        len_row = 6 * num - num
        len_col = 4 * num - num
        if num ==2:
            len_row = len_row + 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(len_row+10, len_col+6))
        fig.suptitle(f'{method} on Residual: {num} Missing Cells {num} vs. Missing Cells Proportion', 
                fontsize=22, y=0.93)  # Adjust the title spacing
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # Initialize an array to store correlations
        correlations = np.zeros((num_rows, num_cols))
        
        # Iterate over cell types and factors
        for i, cell_type in enumerate(proportions[num].columns):
            for j, factor in enumerate(factors[num].columns):
                x = list(proportions[num][cell_type])
                y = list(factors[num][factor])
                # Use the current subplot for plotting
                ax = axes[i, j]
                # Calculate Pearson's correlation coefficient
                r, p = stats.pearsonr(x, y)
                correlations[i, j] = r
                # Map correlation value to color
                color = scalar_map.to_rgba(r)
                # Scatter plot with color based on correlation
                ax.scatter(x, y, c='dimgrey', alpha=0.7)
                ax.set_xlabel(f'{cell_type} Proportions',fontsize=22)
                ax.set_ylabel(f'{fc} {factor}',fontsize=22, labelpad = 0.6)
                ax.set_ylabel(f'{fc} {factor}', fontsize=22, labelpad = 0.6)
                ax.patch.set_facecolor(color)
                ax.patch.set_alpha(1)
                ax.tick_params(axis='both', which='major', labelsize=12)
                #only show RMSE if relevant:
                if rmse_plot:
                    # Calculate RMSE
                    rmse_value = rmse(x, y)
                    ax.annotate('RMSE = {:.2f}'.format(rmse_value),xy=(0.5, 0.9), xycoords='axes fraction',
                            ha='center', va='center', fontsize=18, fontweight = 'bold')         
                ax.annotate('r = {:.2f}'.format(r), xy=(0.5, .82), xycoords='axes fraction',
                                    ha='center', va='center', fontsize=18, fontweight='bold')       
        # Create a colorbar for the scatter plots
        if num > 3:
            bar_pad = 0.02
        else:
            bar_pad =0.01
        cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=bar_pad)
        cax.set_label("Pearson's Correlation (r)", fontsize=18)
        cax.set_alpha(1)
        cax.ax.tick_params(size=3, labelsize=12)

def select_bulks(bulk_type, num_bulks_touse, num_idx_total, res_name, path, bulk_range, rs):

    """
    Helper function to open pseudo-bulk files and select specific bulks for deconvolution analysis.
    
    This function reads multiple pseudo-bulk and proportion files from disk, selects a specific 
    number of bulk samples based on user-defined parameters, and combines the selected data 
    into pandas DataFrames. It allows for the selection of pseudo-bulks based on the type 
    and number of bulks needed, and performs random sampling while ensuring there are no 
    zero proportions in the selected data.

    Args:
        bulk_type (str): The type of bulk to be selected (defined in `bulk_range`).
        num_bulks_touse (int): The number of bulks to randomly select for analysis.
        num_idx_total (int): Total number of indices/pseudo-bulk files to be processed.
        res_name (str): Base name for the pseudo-bulk and proportion files.
        path (str): Directory path where the pseudo-bulk and proportion files are stored.
        bulk_range (dict): Dictionary defining the available indices for each bulk type.
        rs (int): Random seed for reproducibility of bulk selection.

    Returns:
        tuple: Two pandas DataFrames, one containing the selected proportions and the 
               other containing the selected pseudo-bulk data.
    """

    #selecting bulks and props for deconv.
    np.random.seed(rs) 
    prop_df = pd.DataFrame()
    pseudo_df = pd.DataFrame()
    for idx in range(0,num_idx_total):
        #open each patient files:
        pseudobulk_file = os.path.join(path, f"{res_name}_pseudo_{idx}.pkl")
        prop_file = os.path.join(path, f"{res_name}_prop_{idx}.pkl")

        pseudobulk_path = Path(pseudobulk_file)
        prop_path = Path(prop_file)

        prop_df_new = pickle.load( open( prop_path, "rb" ) )
        pseudo_df_new = pickle.load(  open( pseudobulk_path, "rb" ) )
        
        prop_df.index = range(0,len(prop_df))
        pseudo_df.index = range(0,len(pseudo_df))

        #selecting random index withing the desired type
        index_selec = np.random.choice(bulk_range[bulk_type], num_bulks_touse)

        prop_df_new = prop_df_new.iloc[index_selec]
        pseudo_df_new = pseudo_df_new.iloc[index_selec]

        #appending    
        prop_df = prop_df.append(prop_df_new)
        pseudo_df = pseudo_df.append(pseudo_df_new)

        prop_df.index = range(0,len(prop_df))
        pseudo_df.index = range(0,len(pseudo_df))

        if 0 in prop_df.values:
            print("0 in proportion, redo bulk selection")    
    return prop_df, pseudo_df

def calc_nnls(all_refs, prop_df, pseudo_df, num_missing_cells, cells_to_miss):
    """
    Helper function to calculate Non-Negative Least Squares (NNLS) and residuals for deconvolution.

    This function performs NNLS regression to estimate cell type proportions from bulk RNA-seq data 
    and calculates residuals based on the true proportions. It handles scenarios where certain cell types 
    are missing from the reference and compares predicted proportions to the actual proportions in those cases.
    
    The function iterates over different experiments, each with a different number of missing cell types, 
    calculates the proportions using NNLS, rebalances the predicted proportions, and computes residuals for 
    each sample in the dataset.

    Args:
        all_refs (dict): Dictionary of reference datasets with varying missing cell types for each experiment.
        prop_df (pd.DataFrame): DataFrame containing the true cell type proportions for each sample.
        pseudo_df (pd.DataFrame): DataFrame containing pseudo-bulk RNA-seq data for each sample.
        num_missing_cells (list): List of experiments, each corresponding to a number of missing cell types.
        cells_to_miss (dict): Dictionary specifying which cell types to omit for each experiment.

    Returns:
        tuple: Five dictionaries containing:
               - calc_prop_tot: Predicted proportions for each experiment.
               - calc_res_tot: Residuals from the NNLS calculation for each experiment.
               - custom_res_tot: Custom residuals comparing true and predicted proportions.
               - comparison_prop_tot: True proportions with missing cell types removed, for comparison.
               - missing_cell_tot: Proportions of the missing cell types for each experiment.
    """

    calc_prop_tot = dict()
    calc_res_tot = dict()
    custom_res_tot = dict()
    comparison_prop_tot = dict()
    missing_cell_tot =dict()

    for exp in num_missing_cells:
        calc_prop_all = pd.DataFrame()
        custom_res_all = pd.DataFrame()
        calc_res_all  = pd.DataFrame()
        print(f"Exp {exp}")
        #extracting reference with missing cells 
        ref = all_refs[exp].values

        #deleting cell types from the real proportion for comparison
        if exp == 0:
            comparison_prop = prop_df
            missing_cell = []
        else:
            comparison_prop = prop_df.drop(prop_df.columns[cells_to_miss[exp]], axis=1)
            missing_cell = prop_df[prop_df.columns[cells_to_miss[exp]]]

        # calculate predicted values and residuals for each row
        for sample in range(0,len(pseudo_df)):

            #changing ref with exp in num_missing cells
            calc_prop, calc_res = nnls(ref, pseudo_df.iloc[sample].values)
            #putting values in proportion format
            tot = np.sum(calc_prop) #putting them in proportion format
            calc_prop = calc_prop / tot
            #rebalancing the proportions
            prop = comparison_prop.iloc[sample].values #extracting correct proportion
            total_prop = np.sum(prop)
            balanced_prop = prop / total_prop
            #comparison with calculated
            custom_res = balanced_prop - calc_prop

            calc_prop = pd.DataFrame(calc_prop).T
            custom_res = pd.DataFrame(custom_res).T
            
            #putting together
            calc_prop_all = pd.concat([calc_prop_all, calc_prop])
            custom_res_all = pd.concat([custom_res_all, custom_res])
            calc_res_all  = np.append(calc_res_all, calc_res)
            
        #attaching to dicts
        calc_prop_tot[exp] = calc_prop_all
        calc_prop_tot[exp].columns = all_refs[exp].columns
        calc_prop_tot[exp].index = range(0,len(calc_prop_tot[exp]))
        calc_res_tot[exp] = calc_res_all
        custom_res_tot[exp] = custom_res_all
        comparison_prop_tot[exp] = comparison_prop
        missing_cell_tot[exp] = missing_cell

    return calc_prop_tot, calc_res_tot, custom_res_tot, comparison_prop_tot, missing_cell_tot     

def calc_nnls_hgsoc(all_refs, prop_df, pseudo_df, num_missing_cells):
    """
    Helper function to calculate Non-Negative Least Squares (NNLS) and residuals for HGSOC samples.

    This function performs NNLS deconvolution on high-grade serous ovarian cancer (HGSOC) samples
    using reference single-cell RNA-seq data. The goal is to estimate cell type proportions in the bulk RNA-seq
    data (pseudo-bulk) and calculate residuals between the predicted and observed values. The function iterates
    over each pseudo-bulk sample, applying NNLS regression to predict the proportions of each cell type, 
    reformatting the results, and storing them in DataFrames for further analysis.

    Args:
        all_refs (pd.DataFrame): Reference single-cell RNA-seq data for deconvolution.
        prop_df (pd.DataFrame): DataFrame containing the true cell type proportions for each sample.
        pseudo_df (pd.DataFrame): DataFrame containing pseudo-bulk RNA-seq data for each sample.
        num_missing_cells (int): Number of cell types missing in the reference (if applicable).

    Returns:
        tuple: Two pandas DataFrames containing:
               - calc_prop_tot: Predicted cell type proportions for each sample.
               - calc_res_tot: Residuals from the NNLS deconvolution for each sample.
    """

    calc_prop_tot = pd.DataFrame()
    calc_res_tot = pd.DataFrame()
    custom_res_tot = pd.DataFrame()
    comparison_prop_tot = pd.DataFrame()
    missing_cell_tot = pd.DataFrame()

    for idx, bulk in pseudo_df.iterrows():

        calc_prop_all = pd.DataFrame()
        custom_res_all = pd.DataFrame()
        calc_res_all  = pd.DataFrame()

        print(f"Sample No.:{idx}")

        #extracting reference with missing cells 
        ref = all_refs.values

        #using SC reference with each matching bulk
        calc_prop, calc_res = nnls(ref, bulk.values)

        #putting values in proportion format
        tot = np.sum(calc_prop) #putting them in proportion format
        calc_prop = calc_prop / tot

        #putting together
        calc_prop_all = pd.concat([pd.DataFrame(calc_prop_all), pd.DataFrame(calc_prop)])
        calc_res_all  = np.append(calc_res_all, calc_res)
            
        #attaching to dataframes
        calc_prop_tot[idx] = calc_prop_all
        calc_prop_tot[idx].columns = all_refs.columns
        calc_prop_tot[idx].index = range(0,len(calc_prop_tot[idx]))
        calc_res_tot[idx] = calc_res_all


    calc_prop_tot = calc_prop_tot.T
    cals_res_tot = calc_res_tot.T
    calc_prop_tot.columns = all_refs.columns

    return calc_prop_tot, calc_res_tot

def factors_vs_proportions_heatmaps(factors, proportions, num_missing_cells, method):   
    """
    Compares factors (e.g., PCA, SVD, ICA, NMF components) to the missing cell type proportions 
    and visualizes the correlations using scatter plots and heatmaps.

    This function calculates Pearson's correlation coefficient between factor scores from 
    dimensionality reduction methods (PCA, SVD, ICA, NMF) and the cell type proportions for each 
    experiment with varying numbers of missing cell types. It generates scatter plots for visualizing 
    the relationship between factors and proportions and heatmaps to show the correlation values.

    Args:
        factors (dict): A dictionary containing factor scores (components) for each experiment.
        proportions (dict): A dictionary containing the proportions of cell types for each experiment.
        num_missing_cells (list): A list of experiments, each corresponding to a different number of missing cell types.
        method (str): The dimensionality reduction method used ('PCA', 'SVD', 'ICA', or 'NMF').

    Returns:
        None: The function generates scatter plots and heatmaps to visualize the correlations between factors 
              and missing cell type proportions.
    """ 
    if method == "PCA":
        fc = "PC"
    if method == "SVD":
        fc = "SVD"
    if method == "ICA":
        fc = "IC" 
    if method == "NMF":
        fc = "Factor"      
    
    # Create a colormap and normalize object for correlation values
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=-1, vmax=1)
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)

    # Iterate over the number of missing cells
    for num in num_missing_cells[1:]:
        # Define the number of rows and columns for the grid layout
        num_rows = len(proportions[num].columns) 
        num_cols = len(factors[num].columns)
        
        if num_rows == 1:
            # Create a single subplot with two separate scatter plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Two columns for two factors
            fig.suptitle(f'Correlations for {method} on Residual: No. cells missing {num}', fontsize=12)
            x = list(proportions[num].iloc[:, 0])  # Assuming there's only one cell type
            correlations = np.zeros(2)  # Array to store correlations
            
            for j, factor in enumerate(factors[num].columns):
                y = list(factors[num][factor])
                ax = axes[j]  # Use the current subplot for plotting

                # Calculate Pearson's correlation coefficient
                r, p = stats.pearsonr(x, y)
                correlations[j] = r
                
                # Map correlation value to color
                color = scalar_map.to_rgba(r)
                
                # Scatter plot with color based on correlation
                ax.scatter(x, y, c=color)
                ax.set_xlabel(f'{proportions[num].columns[0]} Proportions')
                ax.set_ylabel(f'{fc}{factor}')
                ax.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
                # Add a regression line
                ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='blue')
                
            # Create a colorbar for the scatter plot
            cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist())
            cax.set_label('Correlation (r)')
            
            # Create a heatmap for correlations
            plt.figure(figsize=(8, 4))
            plt.imshow(correlations.reshape(1, -1), cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(label='Correlation (r)')
            plt.title(f'Correlations for {method} on Residual: No. cells missing {num}', fontsize=12)
            plt.xticks(np.arange(2), factors[num].columns)
            plt.yticks([])
            plt.xlabel(f"{fc} No.")
            plt.ylabel("Missing Cell Type")
            plt.ylabel(f'{proportions[num].columns[0]} Proportions')
            plt.tight_layout()
        else:
            # Create a grid of subplots
            len_row = 6 * num - num
            len_col = 4 * num - num
            if num ==2:
                len_row = len_row + 2
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(len_row, len_col))
            fig.suptitle(f'{method} on Residual: No. cells missing {num}, vs. Missing Cell Proportion', fontsize=16)

            # Initialize an array to store correlations
            correlations = np.zeros((num_rows, num_cols))
            
            # Iterate over cell types and factors
            for i, cell_type in enumerate(proportions[num].columns):
                for j, factor in enumerate(factors[num].columns):
                    x = list(proportions[num][cell_type])
                    y = list(factors[num][factor])
                    # Use the current subplot for plotting
                    ax = axes[i, j]
                    
                    # Calculate Pearson's correlation coefficient
                    r, p = stats.pearsonr(x, y)
                    correlations[i, j] = r
                    
                    # Map correlation value to color
                    color = scalar_map.to_rgba(r)
                    
                    # Scatter plot with color based on correlation
                    ax.scatter(x, y, c=color)

                    ax.set_xlabel(f'{cell_type} Proportions')
                    ax.set_ylabel(f'{fc} {factor}')
                    
                    ax.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
                    # Add a regression line
                    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='blue')
            
            # Create a colorbar for the scatter plots
            cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist())
            cax.set_label('Correlation (r)')
            
            # Create a heatmap for correlations
            plt.figure(figsize=(8, 6))
            plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(label='Correlation (r)')
            plt.title(f'Correlations for {method} on Residual: No. cells missing {num}', fontsize=12)
            plt.xticks(np.arange(num_cols), factors[num].columns, rotation=90)
            plt.yticks(np.arange(num_rows), proportions[num].columns)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.xlabel(f"{fc} No.")
            plt.ylabel("Missing Cell Type")
        plt.show()

def make_prop_table(adata, obs):
    """
    Generates a table of cell type proportions from an AnnData object.

    This function calculates the number and proportion of each cell type in an AnnData object based on 
    a specified observation field (e.g., cell type labels). It creates a summary table that includes the 
    total count and proportion of cells, as well as a row for the total number of cells.

    Args:
        adata (AnnData): An AnnData object containing single-cell data.
        obs (str): The name of the observation field in `adata.obs` from which to calculate cell type proportions 
                   (e.g., "cell_types" or any categorical annotation).

    Returns:
        pd.DataFrame: A DataFrame containing three columns:
            - 'Cell_Types': The names of the cell types, with an additional row for the total.
            - 'Num_Cells': The number of cells in each type.
            - 'Prop_Cells': The proportion of cells in each type relative to the total number of cells.
    """
    
    num_cell_counter = Counter(adata.obs[obs])
    num_cells = list()
    cell_types = list()
    prop_cells = list()
    tot_count = 0
    tot_prop = 0

    for cell in num_cell_counter:
        num_cells.append(num_cell_counter[cell])
        cell_types.append(cell)
        tot_count = tot_count + num_cell_counter[cell]

    for cell in num_cell_counter:
        proportion = num_cell_counter[cell] / tot_count
        prop_cells.append(proportion)
        tot_prop = tot_prop + proportion

    cell_types.append('Total')
    num_cells.append(tot_count)
    prop_cells.append(tot_prop)
    table = {'Cell_Types': cell_types, 
        'Num_Cells': num_cells, 
        'Prop_Cells': prop_cells}
    table = pd.DataFrame(table)
    return table        

def factors_vs_proportions(factors, proportions, num_missing_cells, method):
    """
    Compares factors (e.g., PCA, SVD, ICA, NMF components) to missing cell type proportions
    and visualizes the correlations using scatter plots and heatmaps.

    This function calculates Pearson's correlation coefficient between the factor scores 
    (from dimensionality reduction techniques such as PCA, SVD, ICA, or NMF) and the 
    proportions of missing cell types for each experiment. The results are visualized 
    using scatter plots and heatmaps, allowing the user to observe how the factors relate 
    to cell type proportions.

    Args:
        factors (dict): A dictionary containing factor scores (components) for each experiment.
        proportions (dict): A dictionary containing the proportions of cell types for each experiment.
        num_missing_cells (list): A list of experiments, each corresponding to a different number of missing cell types.
        method (str): The dimensionality reduction method used ('PCA', 'SVD', 'ICA', or 'NMF').

    Returns:
        None: The function generates scatter plots and heatmaps to visualize the correlations 
              between factors and missing cell type proportions.
    """
    if method == "PCA":
        fc = "PC"
    if method == "SVD":
        fc = "SVD"
    if method == "ICA":
        fc = "IC" 
    if method == "NMF":
        fc = "Factor"      
   #sample compared to each missing celltype proportion
    # iterate over the number of missing cells
    for num in num_missing_cells[1:]:
        #define the number of rows and columns for the grid layout
        num_rows = len(proportions[num].columns) 
        num_cols = len(factors[num].columns)
        if num_rows == 1:
            #create a single subplot with two separate scatter plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))  #two columns for two factors
            
            x = list(proportions[num].iloc[:, 0])  #assuming there's only one cell type
            correlations = np.zeros(2)  #array to store correlations
            
            for j, factor in enumerate(factors[num].columns):
                y = list(factors[num][factor])
                ax = axes[j]  # use the current subplot for plotting
                ax.scatter(x, y, c='lightslategrey')
                ax.set_xlabel(f'{proportions[num].columns[0]} Proportions')
                ax.set_ylabel(f'{fc}{factor}')
                # Calculate and display Pearson's correlation coefficient
                r, p = stats.pearsonr(x, y)
                ax.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
                # Add a regression line
                ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='blue')
                
                correlations[j] = r
            # create a heatmap for correlations
            plt.figure(figsize=(8, 4))
            plt.imshow(correlations.reshape(1, -1), cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(label='Correlation (r)')
            plt.title(f'Correlations for {method} on Residual: No. cells missing {num}', fontsize=12)
            plt.xticks(np.arange(2), factors[num].columns)
            plt.yticks([])
            plt.xlabel(f"{fc} No.")
            plt.ylabel("Missing Cell Type")
            plt.ylabel(f'{proportions[num].columns[0]} Proportions')
            plt.tight_layout()
        else:
            # create a grid of subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            fig.suptitle(f'{method} on Residual: No. cells missing {num}, vs. Missing Cell Proportion', fontsize=16)

            # initialize an array to store correlations
            correlations = np.zeros((num_rows, num_cols))
            
            # iterate over cell types and factors
            for i, cell_type in enumerate(proportions[num].columns):
                for j, factor in enumerate(factors[num].columns):
                    x = list(proportions[num][cell_type])
                    y = list(factors[num][factor])
                    # use the current subplot for plotting
                    ax = axes[i, j]
                    # scatter plot
                    ax.scatter(x, y, c='lightslategrey')
                    ax.set_xlabel(f'{cell_type} Proportions')
                    ax.set_ylabel(f'{fc} {factor}')
                    
                    #calculate and store Pearson's correlation coefficient
                    r, p = stats.pearsonr(x, y)
                    correlations[i, j] = r
                    ax.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
                    #add a regression line
                    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='blue')
            
            #create a heatmap for correlations
            plt.figure(figsize=(8, 6))
            plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(label='Correlation (r)')
            plt.title(f'Correlations for {method} on Residual: No. cells missing {num}', fontsize=12)
            plt.xticks(np.arange(num_cols), factors[num].columns, rotation=90)
            plt.yticks(np.arange(num_rows), proportions[num].columns)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.xlabel(f"{fc} No.")
            plt.ylabel("Missing Cell Type")
        plt.show()
        #It is expected that only one column (factor) for each cell type (row) will be postivelly correlated.    

def factors_vs_expression(factors, expression, num_missing_cells, method):
    """
    Compares factors (e.g., PCA, SVD, ICA, NMF components) to missing cell type expression levels.

    This function analyzes the relationship between components from a dimensionality reduction 
    technique (such as PCA, SVD, ICA, or NMF) and the expression levels of missing cell types 
    in bulk RNA-seq data. For each experiment where different numbers of cell types are missing, 
    it calculates Pearson's correlation coefficient between the component scores and the expression levels. 
    The results are visualized using scatter plots and heatmaps.

    Args:
        factors (dict): A dictionary containing the factor loadings or components for each experiment.
        expression (dict): A dictionary containing the expression levels of missing cell types for each experiment.
        num_missing_cells (list): A list of experiments, each corresponding to a number of missing cell types.
        method (str): The dimensionality reduction method used ('PCA', 'SVD', 'ICA', or 'NMF').

    Returns:
        None: The function generates scatter plots and heatmaps showing the correlations between factors 
              and expression levels.
    """
    if method == "PCA":
        fc = "PC"
    if method == "SVD":
        fc = "SVD"
    if method == "ICA":
        fc = "IC" 
    if method == "NMF":
        fc = "Factor"      
   #sample compared to each missing celltype expression
    # iterate over the number of missing cells
    # iterate over the number of missing cells
    for num in num_missing_cells[1:]:
        res_PCA_df = factors[num]
        #define the number of rows and columns for the grid layout
        num_rows = len(expression[num].columns) 
        num_cols = len(res_PCA_df.columns)
        
        if num_rows == 1:
            #create a single subplot with two separate scatter plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))  #two columns for two Components
            
            x = list(expression[num].iloc[:, 0])  #assuming there's only one cell type
            correlations = np.zeros(2)  #array to store correlations
            
            for j, Component in enumerate(res_PCA_df.columns):
                y = list(res_PCA_df[Component])
                ax = axes[j]  # use the current subplot for plotting
                ax.scatter(x, y)
                ax.set_xlabel(f'{expression[num].columns[0]} Ref. Expression')
                ax.set_ylabel(f'{fc}{Component}')
                
                # Calculate and display Pearson's correlation coefficient
                r, p = stats.pearsonr(x, y)
                ax.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
                # Add a regression line
                ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
                
                correlations[j] = r
            # create a heatmap for correlations
            plt.figure(figsize=(8, 4))
            plt.imshow(correlations.reshape(1, -1), cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(label='Correlation (r)')
            plt.title(f'Correlations for {method} on Residual: No. cells missing {num}', fontsize=12)
            plt.xticks(np.arange(2), res_PCA_df.columns)
            plt.yticks([])
            plt.xlabel(f"{fc} No.")
            plt.ylabel(f"Missing Cell Type: {expression[num].columns[0]}")
            plt.tight_layout()
        else:
            # create a grid of subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            fig.suptitle(f'{method} on Residual: No. cells missing {num}, vs. Missing Cell Expression', fontsize=16)

            # initialize an array to store correlations
            correlations = np.zeros((num_rows, num_cols))
            
            # iterate over cell types and Components
            for i, cell_type in enumerate(expression[num].columns):
                for j, Component in enumerate(res_PCA_df.columns):
                    x = list(expression[num][cell_type])
                    y = list(res_PCA_df[Component])
                    # use the current subplot for plotting
                    ax = axes[i, j]
                    # scatter plot
                    ax.scatter(x, y)
                    ax.set_xlabel(f'{cell_type} Ref. Expression')
                    ax.set_ylabel(f'{fc}{Component}')
                    
                    #calculate and store Pearson's correlation coefficient
                    r, p = stats.pearsonr(x, y)
                    correlations[i, j] = r
                    ax.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
                    #add a regression line
                    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
            
            #create a heatmap for correlations
            plt.figure(figsize=(8, 6))
            plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(label='Correlation (r)')
            plt.title(f'Correlations for {method} on Residual: No. cells missing {num}', fontsize=12)
            plt.xticks(np.arange(num_cols), res_PCA_df.columns, rotation=90)
            plt.yticks(np.arange(num_rows), expression[num].columns)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.xlabel(f"{fc} No.")
            plt.ylabel("Missing Cell Type")
        plt.show()
    #It is expected that only one column (Component) for each cell type (row) will be postivelly correlated.   

def mean_sqr_error(single1, single2):
    """
    Calculates the mean squared error (MSE) between two arrays.

    This function computes the MSE between two sets of data, `single1` and `single2`, by taking
    the average of the squared differences between corresponding elements. MSE is commonly 
    used to quantify the difference between predicted and observed values.

    Args:
        single1 (array-like): The first array of values.
        single2 (array-like): The second array of values to compare against the first.

    Returns:
        float: The mean squared error between the two arrays.
    """
    return np.mean((single1 - single2)**2)

def get_pert_transform_vec_PCA(X_full, meta_df, curr_samp, fit):
    """
    Computes the perturbation vector in PCA space between stimulated and control samples.

    This function identifies the bulk RNA-seq samples corresponding to a specific perturbation
    (`STIM`) and control condition (`CTRL`), both from the same sample ID. It computes the 
    transformation in the PCA space for both groups and returns the perturbation vector, which 
    is the difference between the median PCA-transformed stimulated and control samples.

    Args:
        X_full (np.ndarray): The full data matrix containing gene expression values for all samples.
        meta_df (pd.DataFrame): A DataFrame containing metadata for each sample, including sample type, 
                                stimulation condition, and whether the sample is used for training.
        curr_samp (str): The sample ID for which to compute the perturbation vector.
        fit (PCA object): A fitted PCA object used to transform the gene expression data.

    Returns:
        np.ndarray: The perturbation vector, which is the difference between the median PCA-transformed
                    values of stimulated and control samples.
    """
    # get the perturbation latent code
    idx_stim_train = np.logical_and(meta_df.samp_type == "bulk", meta_df.isTraining == "Train")
    idx_stim_train = np.logical_and(idx_stim_train, meta_df.stim == "STIM")
    idx_stim_train = np.logical_and(idx_stim_train, meta_df.sample_id == curr_samp)
    idx_stim_train = np.where(idx_stim_train)[0]


    idx_ctrl_train = np.logical_and(meta_df.samp_type == "bulk", meta_df.isTraining == "Train")
    idx_ctrl_train = np.logical_and(idx_ctrl_train, meta_df.stim == "CTRL")
    idx_ctrl_train = np.logical_and(idx_ctrl_train, meta_df.sample_id == curr_samp)
    idx_ctrl_train = np.where(idx_ctrl_train)[0]

    X_ctrl = X_full[idx_ctrl_train]
    train_ctrl = fit.transform(X_ctrl)

    X_stim = X_full[idx_stim_train]
    train_stim = fit.transform(X_stim)


    train_stim_med = np.median(train_stim, axis=0)
    train_ctrl_med = np.median(train_ctrl, axis=0)

    proj_train = train_stim_med - train_ctrl_med
    return(proj_train)

def get_pca_for_plotting(encodings):
    """
    Performs PCA transformation on input encodings and returns the first two principal components.

    This function applies Principal Component Analysis (PCA) to reduce the dimensionality of the
    provided `encodings` to two components, suitable for plotting in 2D space. The result is
    returned as a DataFrame with the two principal components.

    Args:
        encodings (array-like): High-dimensional data to be projected into PCA space.

    Returns:
        pd.DataFrame: A DataFrame containing the first two principal components (PCA_0 and PCA_1).
    """
    from sklearn.decomposition import PCA

    fit = PCA(n_components=2)
    pca_results = fit.fit_transform(encodings)

    plot_df = pd.DataFrame(pca_results[:,0:2])
    print(pca_results.shape)
    print(plot_df.shape)
    plot_df.columns = ['PCA_0', 'PCA_1']
    return plot_df

def plot_pca(plot_df, color_vec, ax, title="", alpha=0.1):
    """
    Plots the PCA projection using Seaborn.

    This function generates a scatter plot of the first two principal components obtained
    from PCA. Points are colored based on the `color_vec`, and the plot is displayed on
    the provided axis (`ax`). The transparency of the points can be controlled using the
    `alpha` parameter.

    Args:
        plot_df (pd.DataFrame): DataFrame containing the PCA coordinates (PCA_0 and PCA_1).
        color_vec (array-like): A vector that defines the color category for each point.
        ax (matplotlib.axes._axes.Axes): Axis on which to plot the PCA projection.
        title (str, optional): Title of the plot. Default is an empty string.
        alpha (float, optional): Transparency level for the points (0 is fully transparent, 1 is fully opaque). Default is 0.1.

    Returns:
        g (seaborn.axisgrid.FacetGrid): The PCA scatter plot.
    """

    plot_df['Y'] = color_vec

    g = sns.scatterplot(
        x="PCA_0", y="PCA_1",
        data=plot_df,
        hue="Y",
        palette=sns.color_palette("hls", len(np.unique(color_vec))),
        legend="full",
        alpha=alpha, ax=ax
    )

    ax.set_title(title)
    return g

def get_tsne_for_plotting(encodings):
    """
    Performs t-SNE transformation on input encodings and returns the 2D t-SNE coordinates.

    This function applies t-distributed Stochastic Neighbor Embedding (t-SNE) to reduce the
    dimensionality of the provided `encodings` to two components for 2D visualization. The
    resulting t-SNE coordinates are returned as a DataFrame with two columns (`tsne_0` and `tsne_1`).

    Args:
        encodings (array-like): High-dimensional data to be projected into t-SNE space.

    Returns:
        pd.DataFrame: A DataFrame containing the 2D t-SNE coordinates (`tsne_0` and `tsne_1`).
    """
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    tsne_results = tsne.fit_transform(encodings)

    plot_df = pd.DataFrame(tsne_results[:, 0:2])
    print(tsne_results.shape)
    print(plot_df.shape)
    plot_df.columns = ['tsne_0', 'tsne_1']
    return plot_df

def plot_tsne(plot_df, color_vec, ax, title=""):
    """
    Plots a t-SNE projection of data using Seaborn.

    This function generates a t-SNE scatter plot of the provided data, with points colored
    based on a categorical vector. The t-SNE coordinates are expected to be stored in 
    `plot_df` under the columns `tsne_0` and `tsne_1`. The color of each point is determined
    by the values in `color_vec`, and the plot is displayed on the provided axis (`ax`).

    Args:
        plot_df (pd.DataFrame): DataFrame containing t-SNE coordinates for plotting.
        color_vec (array-like): A vector that defines the color category for each point.
        ax (matplotlib.axes._axes.Axes): Axis on which to plot the t-SNE projection.
        title (str, optional): Title of the plot. Default is an empty string.

    Returns:
        g (seaborn.axisgrid.FacetGrid): The t-SNE scatter plot.
    """

    plot_df['Y'] = color_vec

    g = sns.scatterplot(
        x="tsne_0", y="tsne_1",
        data=plot_df,
        hue="Y",
        palette=sns.color_palette("hls", len(np.unique(color_vec))),
        legend="full",
        alpha=0.3, ax= ax
    )

    ax.set_title(title)
    return g

def convert_pvalue_to_asterisks(pvalue):
    """
    Converts a p-value to its corresponding significance level represented by asterisks.

    This function returns a string of asterisks to indicate statistical significance based on
    the provided p-value. It uses conventional thresholds for significance:
    - **** for p ≤ 0.0001
    - *** for p ≤ 0.001
    - ** for p ≤ 0.01
    - * for p ≤ 0.05
    - "ns" (not significant) for p > 0.05

    Args:
        pvalue (float): The p-value to be converted.

    Returns:
        str: A string representing the level of statistical significance.
    """

    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def get_prop(adata):
    """
    Extracts cell type proportions from an AnnData object and removes non-frequent cell types.

    This function computes the proportions of each cell type in the provided `adata` object,
    removes rows that correspond to the 'Total' cell type, and returns the proportions sorted 
    by cell type. It also returns the input `adata` object for further analysis.

    Args:
        adata (AnnData): An AnnData object containing single-cell data.

    Returns:
        tuple: 
            - props (pd.DataFrame): A DataFrame containing the sorted cell type proportions.
            - adata (AnnData): The original AnnData object.
    """

    props = make_prop_table(adata, "cell_types")
    idx_total = props[ (props['Cell_Types'] == 'Total')].index
    props = props.drop(idx_total, inplace=False)
    props = props.sort_values(by='Cell_Types')
    props = props.set_index('Cell_Types')
    print(props)
    return props, adata
