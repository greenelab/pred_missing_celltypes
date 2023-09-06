#The following functions come 

# general imports
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Softmax, ReLU, ELU, LeakyReLU
from tensorflow.keras.layers import concatenate as concat
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, KLDivergence
from tensorflow.keras.datasets import mnist
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.utils import to_categorical, normalize, plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
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
import scipy as sp
from scipy.optimize import nnls

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, MinMaxScaler
from matplotlib_venn import venn2
from upsetplot import from_contents, UpSet

# programming stuff
import time
import os
import pickle
from pathlib import Path

#helper function to open pseudo files and selecting some bulks depending on needs
def select_bulks(bulk_type, num_bulks_touse, num_idx_total, res_name, path, bulk_range):
    #selecting bulks and props for deconv.
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

# for each sample calculate the transformation / projection in PCA space
def get_samp_transform_vec_VAE(X_full, meta_df, start_samp, end_samp, encoder, decoder, batch_size):
    # get the perturbation latent code
    idx_start_train = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_start_train = np.logical_and(idx_start_train, meta_df.sample_id == start_samp)
    idx_start_train = np.where(idx_start_train)[0]


    idx_end_train = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_end_train = np.logical_and(idx_end_train, meta_df.sample_id == end_samp)
    idx_end_train = np.where(idx_end_train)[0]
    idx_end_train = np.tile(idx_end_train, 50)

    X_start = X_full[idx_start_train]
    mu_slack, z_slack = encoder.predict(X_start, batch_size=batch_size)
    train_start = decoder.predict(mu_slack, batch_size=batch_size)

    X_end = X_full[idx_end_train]
    mu_slack, z_slack = encoder.predict(X_end, batch_size=batch_size)
    train_end = decoder.predict(mu_slack, batch_size=batch_size)


    train_start_med = np.median(train_start, axis=0)
    train_end_med = np.median(train_end, axis=0)

    proj_train = train_start_med - train_end_med
    return proj_train

#the following functions are adapted from:
# https://github.com/greenelab/sc_bulk_ood/blob/main/method_comparison/validation_plotting.py
def get_pert_transform_vec_VAE(X_full, meta_df, curr_samp, encoder, decoder, batch_size):

    # get the perturbation latent code
    idx_stim_train = np.logical_and(meta_df.samp_type == "bulk", meta_df.isTraining == "Train")
    idx_stim_train = np.logical_and(idx_stim_train, meta_df.stim == "STIM")
    idx_stim_train = np.logical_and(idx_stim_train, meta_df.sample_id == curr_samp)
    idx_stim_train = np.where(idx_stim_train)[0]
    idx_stim_train = np.tile(idx_stim_train, 50)

    idx_ctrl_train = np.logical_and(meta_df.samp_type == "bulk", meta_df.isTraining == "Train")
    idx_ctrl_train = np.logical_and(idx_ctrl_train, meta_df.stim == "CTRL")
    idx_ctrl_train = np.logical_and(idx_ctrl_train, meta_df.sample_id == curr_samp)
    idx_ctrl_train = np.where(idx_ctrl_train)[0]
    idx_ctrl_train = np.tile(idx_ctrl_train, 50)

    X_ctrl = X_full[idx_ctrl_train]
    mu_slack, z_slack = encoder.predict(X_ctrl, batch_size=batch_size)
    train_ctrl = decoder.predict(mu_slack, batch_size=batch_size)

    X_stim = X_full[idx_stim_train]
    mu_slack, z_slack = encoder.predict(X_stim, batch_size=batch_size)
    train_stim = decoder.predict(mu_slack, batch_size=batch_size)


    train_stim_med = np.median(train_stim, axis=0)
    train_ctrl_med = np.median(train_ctrl, axis=0)

    proj_train = train_stim_med - train_ctrl_med

    return proj_train

def calc_VAE_perturbation(X_full, meta_df, encoder, decoder, 
                           scaler, batch_size):
    # get the perturbation latent code
    idx_sc_ref = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.samp_type == "sc_ref")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.cell_prop_type == "cell_type_specific")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.sample_id == "1015")
    idx_sc_ref = np.where(idx_sc_ref)[0]
    sc_ref_meta_df = meta_df.iloc[idx_sc_ref]

    X_sc_ref = np.copy(X_full)
    X_sc_ref = X_sc_ref[idx_sc_ref,]

    ## get the transformation vectors
    proj_samp_dict = {}
    proj_pert_dict = {}
    start_samps = ['1015'] #['1015', '1256']
    end_samps = ['1488', '1244', '1016', '101', '1039', '107']
    for start_samp in start_samps:
        for end_samp in end_samps:
            proj_vec = get_samp_transform_vec_VAE(X_full, meta_df, start_samp, end_samp, encoder, decoder, batch_size)
            proj_samp_dict[f"{start_samp}_{end_samp}"] = proj_vec
    for curr_samp in end_samps:
        proj_vec = get_pert_transform_vec_VAE(X_full, meta_df, curr_samp, encoder, decoder, batch_size)
        proj_pert_dict[curr_samp] = proj_vec


    # get the CTRL
    mu_slack, z_slack = encoder.predict(X_sc_ref, batch_size=batch_size)
    single_decoded_0_0 = decoder.predict(z_slack, batch_size=batch_size)
    single_decoded_0_1 = np.copy(single_decoded_0_0)

    # do the projections
    decoded_0_0 = None
    decoded_0_1 = None
    final_meta_df = None
    for curr_samp_end in end_samps:
        curr_decoded_0_0 = single_decoded_0_0.copy()
        curr_decoded_0_1 = single_decoded_0_1.copy()
        curr_meta_df = sc_ref_meta_df.copy()
        for curr_idx in range(X_sc_ref.shape[0]):
            # project for each initial sample
            curr_samp_start = curr_meta_df.iloc[curr_idx].sample_id
            # project to sample
            proj_samp_vec = proj_samp_dict[f"{curr_samp_start}_{curr_samp_end}"]
            # project to perturbation
            proj_pert_vec = proj_pert_dict[curr_samp_end]

            curr_decoded_0_0[curr_idx] = curr_decoded_0_0[curr_idx] + proj_samp_vec
            curr_decoded_0_1[curr_idx] = curr_decoded_0_0[curr_idx] + proj_pert_vec
            curr_meta_df.iloc[curr_idx].sample_id = curr_samp_end
            curr_meta_df.iloc[curr_idx].isTraining = "Test"

        ### append new df
        if final_meta_df is None:
            decoded_0_0 = curr_decoded_0_0
            decoded_0_1 = curr_decoded_0_1
            final_meta_df = curr_meta_df
        else:
            decoded_0_0 = np.append(decoded_0_0, curr_decoded_0_0, axis=0)
            decoded_0_1 = np.append(decoded_0_1, curr_decoded_0_1, axis=0)
            final_meta_df = final_meta_df.append(curr_meta_df)

    decoded_0_1 = scaler.inverse_transform(decoded_0_1)

    decoded_0_0 = scaler.inverse_transform(decoded_0_0)


    return (final_meta_df, decoded_0_0, decoded_0_1)

def get_samp_transform_vec_PCA(X_full, meta_df, start_samp, end_samp, fit):
    # get the perturbation latent code
    idx_start_train = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_start_train = np.logical_and(idx_start_train, meta_df.sample_id == start_samp)
    idx_start_train = np.where(idx_start_train)[0]


    idx_end_train = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_end_train = np.logical_and(idx_end_train, meta_df.sample_id == end_samp)
    idx_end_train = np.where(idx_end_train)[0]

    X_start = X_full[idx_start_train]
    train_start = fit.transform(X_start)

    X_end = X_full[idx_end_train]
    train_end = fit.transform(X_end)


    train_start_med = np.median(train_start, axis=0)
    train_end_med = np.median(train_end, axis=0)

    proj_train = train_start_med - train_end_med
    return(proj_train)

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

def calc_PCA_perturbation(X_full, meta_df, scaler, fit):

    # get the perturbation latent code
    idx_sc_ref = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.samp_type == "sc_ref")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.cell_prop_type == "cell_type_specific")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.sample_id == "1015")
    idx_sc_ref = np.where(idx_sc_ref)[0]
    sc_ref_meta_df = meta_df.iloc[idx_sc_ref]

    X_sc_ref = np.copy(X_full)
    X_sc_ref = X_sc_ref[idx_sc_ref,]

    ## get the transofrmation vectors
    proj_samp_dict = {}
    proj_pert_dict = {}
    start_samps = ['1015'] #['1015', '1256']
    end_samps = ['1488', '1244', '1016', '101', '1039', '107']
    for start_samp in start_samps:
        for end_samp in end_samps:
            proj_vec = get_samp_transform_vec_PCA(X_full, meta_df, start_samp, end_samp, fit)
            proj_samp_dict[f"{start_samp}_{end_samp}"] = proj_vec
    for curr_samp in end_samps:
        proj_vec = get_pert_transform_vec_PCA(X_full, meta_df, curr_samp, fit)
        proj_pert_dict[curr_samp] = proj_vec


    # now get the refernce sample that we will use to do all projectsions
    single_decoded_0_0 = fit.transform(X_sc_ref)
    single_decoded_0_1 = np.copy(single_decoded_0_0)

    # do the projections
    decoded_0_0 = None
    decoded_0_1 = None
    final_meta_df = None
    for curr_samp_end in end_samps:
        curr_decoded_0_0 = single_decoded_0_0.copy()
        curr_decoded_0_1 = single_decoded_0_1.copy()
        curr_meta_df = sc_ref_meta_df.copy()
        for curr_idx in range(X_sc_ref.shape[0]):
            # project for each initial sample
            curr_samp_start = curr_meta_df.iloc[curr_idx].sample_id
            # project to sample
            proj_samp_vec = proj_samp_dict[f"{curr_samp_start}_{curr_samp_end}"]
            # project to perturbation
            proj_pert_vec = proj_pert_dict[curr_samp_end]

            curr_decoded_0_0[curr_idx] = curr_decoded_0_0[curr_idx] + proj_samp_vec
            curr_decoded_0_1[curr_idx] = curr_decoded_0_0[curr_idx] + proj_pert_vec
            curr_meta_df.iloc[curr_idx].sample_id = curr_samp_end
            curr_meta_df.iloc[curr_idx].isTraining = "Test"

        ### append new df
        if final_meta_df is None:
            decoded_0_0 = curr_decoded_0_0
            decoded_0_1 = curr_decoded_0_1
            final_meta_df = curr_meta_df
        else:
            decoded_0_0 = np.append(decoded_0_0, curr_decoded_0_0, axis=0)
            decoded_0_1 = np.append(decoded_0_1, curr_decoded_0_1, axis=0)
            final_meta_df = final_meta_df.append(curr_meta_df)


    decoded_0_1 = fit.inverse_transform(decoded_0_1)
    decoded_0_1 = scaler.inverse_transform(decoded_0_1)

    decoded_0_0 = fit.inverse_transform(decoded_0_0)
    decoded_0_0 = scaler.inverse_transform(decoded_0_0)

    return (final_meta_df, decoded_0_0, decoded_0_1)

def calc_CVAE_perturbation(X_full, meta_df, encoder, decoder, 
                           scaler, batch_size, 
                           label_1hot_full, drug_1hot_full):

    label_1hot_temp = np.copy(label_1hot_full)
    perturb_1hot_temp = np.copy(drug_1hot_full)


    # get the single cell data 
    idx_sc_ref = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.samp_type == "sc_ref")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.cell_prop_type == "cell_type_specific")
    idx_sc_ref = np.where(idx_sc_ref)[0]

    ## this is to match up sample amounts across comparators
    idx_sc_ref = np.tile(idx_sc_ref, 6) 


    X_sc_ref = np.copy(X_full)
    X_sc_ref = X_sc_ref[idx_sc_ref,]

    # get the sample_ids we will perturb
    sample_interest = ['1488', '1244', '1016', '101', '1039', '107']
    sample_code_idx = np.logical_and(meta_df.cell_prop_type == "cell_type_specific", 
                                        np.isin(meta_df.sample_id, sample_interest))
    sample_code_idx = np.where(sample_code_idx)[0]
    sample_code = label_1hot_temp[sample_code_idx]

    # make the metadata file
    ctrl_test_meta_df = meta_df.copy()
    ctrl_test_meta_df = ctrl_test_meta_df.iloc[sample_code_idx]
    ctrl_test_meta_df.isTraining = "Test"
    ctrl_test_meta_df.stim = "CTRL"

    # get the perturb code
    idx_stim = np.where(meta_df.stim == "STIM")[0][range(6000)]
    idx_stim = np.tile(idx_stim, 2)
    perturbed_code = perturb_1hot_temp[idx_stim]

    idx_ctrl = np.where(meta_df.stim == "CTRL")[0][range(6000)]
    idx_ctrl = np.tile(idx_ctrl, 2)
    unperturbed_code = perturb_1hot_temp[idx_ctrl]


    mu_slack = encoder.predict([X_sc_ref, sample_code, perturbed_code], batch_size=batch_size)
    z_concat = np.hstack([mu_slack, sample_code, perturbed_code])
    decoded_0_1 = decoder.predict(z_concat, batch_size=batch_size)
    decoded_0_1 = scaler.inverse_transform(decoded_0_1)


    mu_slack = encoder.predict([X_sc_ref, sample_code, unperturbed_code], batch_size=batch_size)
    z_concat = np.hstack([mu_slack, sample_code, unperturbed_code])
    decoded_0_0 = decoder.predict(z_concat, batch_size=batch_size)
    decoded_0_0 = scaler.inverse_transform(decoded_0_0)

    return (ctrl_test_meta_df, decoded_0_0, decoded_0_1)

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

def get_tsne_for_plotting(encodings):
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=500)
    tsne_results = tsne.fit_transform(encodings)

    plot_df = pd.DataFrame(tsne_results[:,0:2])
    print(tsne_results.shape)
    print(plot_df.shape)
    plot_df.columns = ['tsne_0', 'tsne_1']
    return plot_df

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

import umap
def get_umap_for_plotting(encodings):
    fit = umap.UMAP()
    umap_results = fit.fit_transform(encodings)

    plot_df = pd.DataFrame(umap_results[:,0:2])
    print(umap_results.shape)
    print(plot_df.shape)
    plot_df.columns = ['umap_0', 'umap_1']
    return plot_df

def plot_umap(plot_df, color_vec, ax, title="", alpha=0.3):

    plot_df['Y'] = color_vec

    g = sns.scatterplot(
        x="umap_0", y="umap_1",
        data=plot_df,
        hue="Y",
        palette=sns.color_palette("hls", len(np.unique(color_vec))),
        legend="full",
        alpha=alpha, ax= ax
    )

    ax.set_title(title)
    return g

def plot_expr_corr(xval, yval, ax, title, xlab, ylab, class_id, max_val=2700, min_val=0, alpha=0.5):

    plot_df = pd.DataFrame(list(zip(xval, yval)))
    plot_df.columns = [xlab, ylab]

    g = sns.scatterplot(
        x=xlab, y=ylab,
        data=plot_df,ax=ax,
        hue=class_id,
        alpha= alpha
    )
    g.set(ylim=(min_val, max_val))
    g.set(xlim=(min_val, max_val))
    g.plot([min_val, max_val], [min_val, max_val], transform=g.transAxes)


    ax.set_title(title)
    return g

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

#funtion from https://github.com/greenelab/sc_bulk_ood/blob/main/evaluation_experiments/pbmc/pbmc_experiment_perturbation.ipynb
def mean_sqr_error(single1, single2):
  return np.mean((single1 - single2)**2)    