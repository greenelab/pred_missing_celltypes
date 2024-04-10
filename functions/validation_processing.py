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
# programming stuff
import time
import os
import pickle
from pathlib import Path

def capitalize_first_letters(strings):
    '''Capitalize the first letter of array of strings'''
    return [s.capitalize() for s in strings]

#funct to calculate RMSE
def rmse(y, y_pred):
    # Ensure both y and y_pred are 2D arrays with the same shape
    y = np.array(y).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    
    # Calculate RMSE
    return np.sqrt(((y - y_pred)**2).mean())

def factors_vs_proportions_rmse(factors, proportions, num_missing_cells, method):
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
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Two columns for two factors
            fig.suptitle(f'{method} on Residual: {num} Missing Cell {num} vs. Missing Cell Proportion', fontsize=16,y=0.95)
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
                rmse_value = rmse(x, y)#, squared=False)
                # Map correlation value to color
                color = scalar_map.to_rgba(r)

                # Scatter plot with color based on correlation
                ax.scatter(x, y, c='dimgrey', alpha=0.7)
                ax.set_xlabel(f'{proportions[num].columns[0]} Proportions', fontsize=12)
                ax.set_ylabel(f'{fc}{factor}', fontsize=12, labelpad = 0.5)
                ax.patch.set_facecolor(color)
                ax.patch.set_alpha(1)
                ax.annotate('RMSE = {:.2f}'.format(rmse_value),xy=(0.5, 0.9), xycoords='axes fraction',
                        ha='center', va='center', fontsize=10, fontweight = 'bold')
                
            # Create a colorbar for the scatter plot
            cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=0.01)
            cax.set_label('Correlation (r)', fontsize=12)
            cax.set_alpha(0.4)
        else:
            # Create a grid of subplots
            len_row = 6 * num - num
            len_col = 4 * num - num
            if num ==2:
                len_row = len_row + 2
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(len_row, len_col))
            fig.suptitle(f'{method} on Residual: {num} Missing Cells {num} vs. Missing Cells Proportion', 
                    fontsize=16, y=0.93)  # Adjust the title spacing
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
                    rmse_value = rmse(x, y)#, squared=False)
                    # Scatter plot with color based on correlation
                    ax.scatter(x, y, c='dimgrey', alpha=0.7)
                    if len(cell_type) < 8:
                        ax.set_xlabel(f'{cell_type} Proportions',fontsize=12)
                    else:
                        #adding new line
                        ax.set_xlabel(f'{cell_type}\nProportions', fontsize=12)
                    ax.set_ylabel(f'{fc} {factor}', fontsize=12, labelpad = 0.5)
                    ax.patch.set_facecolor(color)
                    ax.patch.set_alpha(1)
                    ax.annotate('RMSE = {:.2f}'.format(rmse_value), xy=(0.5, 0.9), xycoords='axes fraction', 
                            ha='center', va='center', fontsize=10, fontweight="bold")

            # Create a colorbar for the scatter plots
            if num > 3:
                bar_pad = 0.02
            else:
                bar_pad =0.01
            cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=bar_pad)
            cax.set_label('Correlation (r)', fontsize=12)
            cax.set_alpha(1)

# Function to compare factors to the missing cell type proportions, adapted for real data
def factors_vs_proportions_heatmaps_real(factors, proportions, num, method, rmse_plot):
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Two columns for two factors
        fig.suptitle(f'{method} on Residual: {num} Missing Cell {num} vs. Missing Cell Proportion', fontsize=16,y=0.95)
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
            # Map correlation value to color
            color = scalar_map.to_rgba(r)
            
            # Scatter plot with color based on correlation
            ax.scatter(x, y, c='dimgrey', alpha=0.7)
            if len(cell_type) < 8:
                ax.set_xlabel(f'{cell_type} Proportions',fontsize=12)
            else:
                formatted_label = '\n'.join(cell_type.split())   
                ax.set_xlabel(f'{formatted_label} Proportions',fontsize=12)
            ax.set_ylabel(f'{fc} {factor}', fontsize=12, labelpad = 0.5)
            ax.set_xlabel(f'{proportions[num].columns[0]} Proportions', fontsize=12)
            ax.patch.set_facecolor(color)
            ax.patch.set_alpha(1)
            #only show RMSE if relevant:
            if rmse_plot:
                # Calculate RMSE
                rmse_value = rmse(x, y)
                ax.annotate('RMSE = {:.2f}'.format(rmse_value),xy=(0.5, 0.9), xycoords='axes fraction',
                        ha='center', va='center', fontsize=10, fontweight = 'bold')
        # Create a colorbar for the scatter plot
        cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=0.01)
        cax.set_label('Correlation (r)', fontsize=12)
        cax.set_alpha(0.4)
    else:
        # Create a grid of subplots
        len_row = 6 * num - num
        len_col = 4 * num - num
        if num ==2:
            len_row = len_row + 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(len_row, len_col))
        fig.suptitle(f'{method} on Residual: {num} Missing Cells {num} vs. Missing Cells Proportion', 
                fontsize=16, y=0.93)  # Adjust the title spacing
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

                # Scatter plot with color based on correlation
                ax.scatter(x, y, c='dimgrey', alpha=0.7)
                ax.set_xlabel(f'{cell_type} Proportions',fontsize=12)
                ax.set_ylabel(f'{fc} {factor}',fontsize=12)
                ax.set_ylabel(f'{fc} {factor}', fontsize=12)
                ax.patch.set_facecolor(color)
                ax.patch.set_alpha(1)
                #only show RMSE if relevant:
                if rmse_plot:
                    # Calculate RMSE
                    rmse_value = rmse(x, y)
                    ax.annotate('RMSE = {:.2f}'.format(rmse_value),xy=(0.5, 0.9), xycoords='axes fraction',
                            ha='center', va='center', fontsize=10, fontweight = 'bold')                
        # Create a colorbar for the scatter plots
        if num > 3:
            bar_pad = 0.02
        else:
            bar_pad =0.01
        cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=bar_pad)
        cax.set_label('Correlation (r)', fontsize=12)
        cax.set_alpha(1)

# Helper function to open pseudo files and selecting some bulks depending on needs
def select_bulks(bulk_type, num_bulks_touse, num_idx_total, res_name, path, bulk_range, rs):
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

#helper function to calculate nnls and residual
def calc_nnls(all_refs, prop_df, pseudo_df, num_missing_cells, cells_to_miss):
    
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

# Function to compare factors to the missing cell type proportions
def factors_vs_proportions_heatmaps(factors, proportions, num_missing_cells, method):   
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

#Fcn to make table of cell proportions
def make_prop_table(adata, obs):
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

#Fcn to compare factors to the missing cell type proportions
def factors_vs_proportions(factors, proportions, num_missing_cells, method):
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

#Fcn to compare factors to the missing cell type expression
def factors_vs_expression(factors, expression, num_missing_cells, method):
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

#funtion from https://github.com/greenelab/sc_bulk_ood/blob/main/evaluation_experiments/pbmc/pbmc_experiment_perturbation.ipynb
def mean_sqr_error(single1, single2):
  return np.mean((single1 - single2)**2)    

#the following functions are adapted from:
# https://github.com/greenelab/sc_bulk_ood/blob/main/method_comparison/validation_plotting.py
def get_pert_transform_vec_PCA(X_full, meta_df, curr_samp, fit):
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

# for each sample calculate the transformation / projection in PCA space
def get_pca_for_plotting(encodings):

    from sklearn.decomposition import PCA

    fit = PCA(n_components=2)
    pca_results = fit.fit_transform(encodings)

    plot_df = pd.DataFrame(pca_results[:,0:2])
    print(pca_results.shape)
    print(plot_df.shape)
    plot_df.columns = ['PCA_0', 'PCA_1']
    return plot_df

def plot_pca(plot_df, color_vec, ax, title="", alpha=0.1):

    plot_df['Y'] = color_vec

    g = sns.scatterplot(
        x="PCA_0", y="PCA_1",
        data=plot_df,
        hue="Y",
        palette=sns.color_palette("hls", len(np.unique(color_vec))),
        legend="full",
        alpha=alpha, ax= ax
    )

    ax.set_title(title)
    return g

#get tsne projection
def get_tsne_for_plotting(encodings):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    tsne_results = tsne.fit_transform(encodings)

    plot_df = pd.DataFrame(tsne_results[:,0:2])
    print(tsne_results.shape)
    print(plot_df.shape)
    plot_df.columns = ['tsne_0', 'tsne_1']
    return plot_df

#plot tsne projection
def plot_tsne(plot_df, color_vec, ax, title=""):

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


#Funct. adapted from:
# https://bbquercus.medium.com/adding-statistical-significance-asterisks-to-seaborn-plots-9c8317383235
#to convert p value to number of asterisks in plots
def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

#fcn to extract proportions from each adata and remove cell types that are not frequent
def get_prop(adata):
    props = make_prop_table(adata, "cell_types")
    idx_total = props[ (props['Cell_Types'] == 'Total')].index
    props = props.drop(idx_total, inplace=False)
    props = props.sort_values(by='Cell_Types')
    props = props.set_index('Cell_Types')
    print(props)
    return props, adata    