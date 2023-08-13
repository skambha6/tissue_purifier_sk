#!/usr/bin/env python

# This script performs gene expression regression with learned SSL features and compares to NCV 
# change to work with list of anndata? or directory of anndata 

import argparse
import torch
import sys
from typing import List
from anndata import read_h5ad
from tissue_purifier.data import AnndataFolderDM
from tissue_purifier.models.ssl_models import *

import numpy
import numpy as np
import torch
import seaborn
import tarfile
import os
import matplotlib
import matplotlib.pyplot as plt
from anndata import read_h5ad
import scanpy as sc
import pandas as pd
import gseapy as gp

import pdb

# tissue_purifier import
import tissue_purifier as tp

from tissue_purifier.genex import *
from sklearn.linear_model import RidgeCV
    
    
## stratified by majority cell type label
def regress_and_plot(adata, config_dict_):
        
    ### repeat with cell type proportions instead of majority cell type labels

    category_key = config_dict_['category_key']
    
    cell_types = np.array(adata.obsm[category_key].columns)
    
    # loop over all ncvs
    
    covars = config_dict_['feature_keys']

    
    ncols = 10
    #ncols = len(cell_types)
    nmax = len(cell_types)
    nrows = int(numpy.ceil(float(nmax)/ncols))
    
    
    #nrows = 2
    #nrows = int(numpy.ceil(float(nmax)/ncols))

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows))
    
    
    fig.suptitle("'GEX Evaluation, 100 most highly variable genes")

    for i in range(len(cell_types)):
        ## baseline model:

        counts_ng = adata.X.todense()
        
        

        ## make this user parameters
        
        X = np.array(adata.obsm[category_key])
        
        majority_cell_type = np.argmax(X, axis = 1)
        adata.obs['majority_cell_type'] = majority_cell_type

        n_cell_types = len(cell_types)
        
        majority_cell_types = np.argmax(X, axis = 0)

        score = []
        print_score = []
        alpha = []

        r,c = i//ncols, i%ncols
        
        #r,c = 1,1
        
        if len(axes.shape) > 1:
            ax_cur = axes[r,c]
        else:
            ax_cur = axes[c]

        adata_kg = adata[adata.obs['majority_cell_type'] == i]
        keys = ['ct_p']
        
        
        if adata_kg.X.shape[0] > 5: #min # of cells of that cell type to be present at majority level to do analysis
            #print(idx)

            #print(adata_cg)
            #print(len(idx))

            
            X = np.array(adata_kg.obsm[category_key])
            
            counts_kg = adata_kg.X.todense()
            y = counts_kg

            ## train-test split?
            
            clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]).fit(X, y)
            
            if config_dict_["metric"] == "r^2":
                new_score = clf.score(X,y)
            elif config_dict_["metric"] == "q_ratio":
                
                y_hat = clf.predict(X)

                qdist = np.abs(y_hat - y)
                
                baseline_qdist = qdist ## dealing with 0s here? (bc na when divide)
                
                baseline_qdist[baseline_qdist == 0] = 0.001
                
                q_ratio = np.array(qdist/baseline_qdist)
                new_score = np.median(q_ratio)
                
                
                ## create reference z scores etc. 
#                 ## get z-scores 
        
#                 q_ref_mu_kg = np.load('q_ref_mu_kg.npy')
#                 q_ref_std_kg = np.load('q_ref_std_kg.npy')


#                 q_z = (q_kg - q_ref_mu_kg)/q_ref_std_kg
#                 q_z_mean = q_z.mean()
        
            #elif config_dict_["metric"] == "z-score":
                
                
            score.append(new_score)
            
            alpha.append(clf.alpha_)
            print_score.append(round(new_score, 4))

            for n, covar in enumerate(covars):
#                 ## covar model
#                 X = adata_kg.obsm[covar]

#                 y = counts_kg
#                 clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]).fit(X, y)
#                 score.append(clf.score(X,y))
#                 alpha.append(clf.alpha_)
#                 print_score.append(round(clf.score(X,y), 4))

                ## both

                X = np.array(adata_kg.obsm[category_key])
            
                X = np.concatenate([X, adata_kg.obsm[covar]], axis=1)

                y = counts_kg

                clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]).fit(X, y) ## add lasso CV
                
                if config_dict_["metric"] == "r^2":
                    new_score = clf.score(X,y)
                elif config_dict_["metric"] == "q_ratio":
                
                    y_hat = clf.predict(X)

                    qdist = np.abs(y_hat - y)

                    q_ratio = np.array(qdist/baseline_qdist)
                    new_score = np.median(q_ratio)
                    

                    print(cell_types[i])
                    print(covar)
                    
                    if cell_types[i] == "ES" and covar == "dino":
                        run_gsea(q_ratio)
                        
                
                score.append(new_score)
                alpha.append(clf.alpha_)
                print_score.append(round(new_score, 4))
                
                new_key = 'ct_p + ' + covar
                keys.append(new_key)
                

            plt.figure()
            
            container = ax_cur.bar(keys, score)
            ax_cur.bar_label(container, print_score)

            ax_cur.set_ylabel(config_dict_["metric"])
            if config_dict_["metric"] == "r^2":
                ax_cur.set_ylim([0,1.0])
                
            ax_cur.set_title(cell_types[i])
        
    fig.savefig(config_dict_["plot_out"])
    fig.show()
    
def run_gsea(metric_kg):
    ### run GSEA

    qratio_g = np.mean(metric_kg, axis=0)
    qratio_g_pd = pd.DataFrame(qratio_g)
    qratio_g_pd.index = adata.var_names
    
    qratio_g_sorted = qratio_g_pd.sort_values(by=0)
    
    import pdb; pdb.set_trace()
        
    print(qratio_g_sorted.head(n=10))


    pre_res = gp.prerank(rnk=qratio_g_pd, # or rnk = rnk,
                         gene_sets='/mnt/disks/dev/data/m5.go.v2022.1.Mm.symbols.gmt', ## add this as parameter 
                         threads=4,
                         min_size=5,
                         max_size=10000,
                         permutation_num=1000, # reduce number to speed up testing
                         outdir=None, # don't write to disk
                         seed=6,
                         verbose=True, # see what's going on behind the scenes
                        )

    pre_res.res2d.tail(10)
    
    pre_res.res2d.sort_values(by='NES', ascending = False, inplace=True)
    print(pre_res.res2d.head(15)[['Term', 'NES', 'FDR q-val', 'Lead_genes']]) ## write this to out_file
    # from gseapy import gseaplot
    # gseaplot(rank_metric=pre_res.ranking,
    #          term=terms[i],
    #          **pre_res.results[terms[i]])


def parse_args(argv: List[str]) -> dict:
    """
    Read argv from command-line and produce a configuration dictionary.
    If the command-line arguments include

    If the command-line arguments include '--to_yaml my_yaml_file.yaml' the configuration dictionary is written to file.

    Args:
        argv: the parameter passed from the command line.

    Note:
        If argv includes '--config input.yaml' the parameters are read from file.
        The config.yaml parameters have priority over the CLI parameters.

    Note:
        If argv includes '--to_yaml output.yaml' the configuration dictionary is written to file.

    Note:
        Parameters which are missing from both argv and config.yaml will be set to their default values.

    Returns:
        config_dict: a dictionary with all the configuration parameters.
    """
    parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')

    parser.add_argument("--anndata_in", type=str, required=True,
                        help="path to the annotated anndata.h5ad")
    
    parser.add_argument("--plot_out", type=str, required=True,
                        help="Output file name to save images/plots to.")
    
    parser.add_argument("--feature_keys", type=str, nargs='*', required=True,
                        help="The computed features to regress on.")
    
    parser.add_argument("--gsea_out", type=str, required=False,
                        help="Output file name to save gsea results to.")
    
    parser.add_argument("--top_svg", type=str, required=False,
                        help="Output file name to save top svgs to.")
    
    parser.add_argument("--metric", type=str, required=False,
                        help="Metric to evaluate regression with", default="r^2")
                             
    parser.add_argument("--category_key", type=str, required=False,
                        help="Key in obsm containing categories", default="rctd_doublet_weights")
    
    # Add help at the very end
    parser = argparse.ArgumentParser(parents=[parser], add_help=True)

    # Process everything and check
    args = parser.parse_args(argv)
    
    return vars(args)


if __name__ == '__main__':
    config_dict_ = parse_args(sys.argv[1:])

    adata = read_h5ad(config_dict_["anndata_in"])
    
    ## make these user parameters
    
    # filter cells parameters
    fc_bc_min_umi = 200                  # filter cells with too few UMI
    fc_bc_max_umi = 3000                 # filter cells with too many UMI
    fc_bc_min_n_genes_by_counts = 10     # filter cells with too few GENES
    fc_bc_max_n_genes_by_counts = 2500   # filter cells with too many GENES
    fc_bc_max_pct_counts_mt = 5          # filter cells with mitocrondial fraction too high

    # filter genes parameters
    fg_bc_min_cells_by_counts = 3000      # filter genes which appear in too few CELLS
    fg_bc_high_var = 1000                  # filter genes to top n highly variable genes

    # filter rare cell types parameters
    fctype_bc_min_cells_absolute = 100   # filter cell-types which are too RARE in absolute number
    fctype_bc_min_cells_frequency = 0.01 # filter cell-types which are too RARE in relative abundance
    
    ## skip cell_type filtering for now
    
    cell_type_key = "cell_type"

    # mitocondria metrics
    adata.var['mt'] = adata.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # counts cells frequency
    # tmp = adata.obs[cell_type_key].values.describe()
    # print(tmp)
    # mask1 = (tmp["counts"] > fctype_bc_min_cells_absolute)
    # mask2 = (tmp["freqs"] > fctype_bc_min_cells_frequency)
    # mask = mask1 * mask2
    # cell_type_keep = set(tmp[mask].index.values)
    # adata.obs["keep_ctype"] = adata.obs["cell_type"].apply(lambda x: x in cell_type_keep)

    
    adata = adata[adata.obs["total_counts"] > fc_bc_min_umi, :] 
    adata = adata[adata.obs["total_counts"] < fc_bc_max_umi, :] 
    adata = adata[adata.obs["n_genes_by_counts"] > fc_bc_min_n_genes_by_counts, :] 
    adata = adata[adata.obs["n_genes_by_counts"] < fc_bc_max_n_genes_by_counts, :] 
    adata = adata[adata.obs["pct_counts_mt"] < fc_bc_max_pct_counts_mt, :]
    #adata = adata[adata.obs["keep_ctype"] == True, :]
    adata = adata[:, adata.var["n_cells_by_counts"] > fg_bc_min_cells_by_counts]


    ## identify fg_bc_var most highly variable genes using Seurat method (Satija et al. (2015))
    # seurat v1 expects logarithmized data 

    adata_copy = adata.copy()
    
    print(adata_copy)
    
    #sc.pp.normalize_total(adata_copy)
    adata_log = sc.pp.log1p(adata_copy, copy=True)
    sc.pp.highly_variable_genes(adata_log, subset=True, n_top_genes = fg_bc_high_var, flavor='seurat')
    adata = adata_log
    
    ## evaluate features using linear regression
    
    regress_and_plot(adata, config_dict_)
    
    # print('metric_kg shape')
    # print(metric_kg.shape)
    # if metric_kg is not None:
    #     run_gsea(metric_kg)
    
    ## get k top SVGs
    
    ## perform GSEA 
    

    
    
                        
                        
                        
## deprecated
def regress_and_plot_separate(adata, config_dict_):
        ### repeat with cell type proportions instead of majority cell type labels

    # loop over all ncvs

    #covars = ['ncv_k10', 'ncv_k25', 'ncv_k100', 'simclr_block-neg', 'simclr_no-block', 'barlow']
    covars = adata.obsm

    nmax = len(covars)
    ncols = 2
    nrows = 2
    #nrows = int(numpy.ceil(float(nmax)/ncols))

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows))
    fig.suptitle("'GEX Evaluation, 100 most highly variable genes")


    for n, covar in enumerate(covars):

        r,c = n//ncols, n%ncols
        #r,c = 1,1
        ax_cur = axes[r,c]

        score = []
        alpha = []
        print_score = []

        ## baseline model:

        counts_ng = adata.X.todense()

        ## make this user parameters
        
        X = np.array(adata.obs[['ES', 'RS', 'Myoid', 'SPC', 'SPG', 'Sertoli', 'Leydig', 'Endothelial', 'Macrophage']])

        y = counts_ng

        clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]).fit(X, y)
        score.append(clf.score(X,y))
        alpha.append(clf.alpha_)
        print_score.append(round(clf.score(X,y), 4))

        ## covar model
        X = adata.obsm[covar]

        y = counts_ng
        clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]).fit(X, y)
        score.append(clf.score(X,y))
        alpha.append(clf.alpha_)
        print_score.append(round(clf.score(X,y), 4))

        ## both

        X = np.array(adata.obs[['ES', 'RS', 'Myoid', 'SPC', 'SPG', 'Sertoli', 'Leydig', 'Endothelial', 'Macrophage']])

        X = np.concatenate([X, adata.obsm[covar]], axis=1)

        y = counts_ng

        clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]).fit(X, y)
        score.append(clf.score(X,y))
        alpha.append(clf.alpha_)
        print_score.append(round(clf.score(X,y), 4))

        plt.figure()
        keys = ['cell_type_prop', covar, 'cell_type_prop + ' + covar]

        container = ax_cur.bar(keys, score)
        ax_cur.bar_label(container, print_score)
        
        ax_cur.set_ylabel('Coefficient of Determination')
        ax_cur.set_ylim([0,0.2])
        ax_cur.set_title(covar)
        
    fig.savefig(config_dict_["out_file"])
    fig.show()