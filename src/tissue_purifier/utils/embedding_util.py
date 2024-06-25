
from typing import Union, Tuple, List
import anndata
# import anndata as ad
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV, RidgeCV

from tissue_purifier.utils import *
from tissue_purifier.plots import *

from tissue_purifier.utils.anndata_util import *

# Set of functions to perform common desired tasks with embeddings, 
# i.e. motif query, clustering, and conditional motif enrichment

def motif_query(adata_ref: anndata.AnnData,
                adata_query: anndata.AnnData,
                query_point: tuple,
                rep_key: str,
                dist_type: str = 'cosine',
                exponentiate: bool=True ## used for cosine distance
                ):

    x_query, y_query = query_point
    query_point_idx = np.argmin((adata_ref.obs['y'].values - x_query) ** 2 + (adata_ref.obs['x'].values - y_query) **2)

    ## Compute similarity of query patch to all patches in reference sample

    query_z = adata_query.obsm[rep_key][query_point_idx]

    if dist_type == 'euclidean':
        dist_n = np.linalg.norm(adata_ref.obsm[rep_key] - query_z[None, :], axis=-1)
        sim_n = np.exp(-dist_n / (2.0 * np.std(dist_n)))

    elif dist_type == 'cosine':
        sim_n = np.sum(adata_ref.obsm[rep_key] * query_z[None, :], -1) / (np.linalg.norm(adata_ref.obsm[rep_key], axis=-1) * np.linalg.norm(query_z))
        if exponentiate:
            sim_n = np.clip(sim_n, a_min=0., a_max=1.) ** 5
        else:
            sim_n = np.clip(sim_n, a_min=0., a_max=1.)
        sim_n[np.where(np.isnan(sim_n))[0]] = 0
        
    adata_ref.obs['sim'] = sim_n

    return adata_ref


def cluster(adata: Union[anndata.AnnData, List[anndata.AnnData]],
               key: str,
               leiden_res: Union[float, list],
               n_neighbors: int = 500,
               umap_preprocess_strategy: str = 'raw',
               cluster_patches: bool = True):
    
    if cluster_patches:
        assert 'sparse_image_state_dict' in adata.uns.keys(), \
                'Need to have sparse_image_state_dict in adata.uns to compute patch clusters'
        pass
        # to be implemented
    else:
        assert isinstance(adata, anndata.AnnData), 'Need to pass an anndata object to cluster'
        ## Compute UMAP 
            
        print("Running UMAP")
        smart_pca = SmartPca(preprocess_strategy='z_score')
        smart_umap = SmartUmap(n_neighbors=n_neighbors, preprocess_strategy=umap_preprocess_strategy, n_components=2, min_dist=0.5, metric='euclidean')
                
        input_features = adata.obsm[key]
        embeddings_pca = smart_pca.fit_transform(input_features, n_components=0.9)
        embeddings_umap = smart_umap.fit_transform(embeddings_pca)
            
        # adata.obsm["pca_"+key] = np.array(embeddings_pca)
        adata.obsm["umap_"+key] = np.array(embeddings_umap)
        adata.obsm["umap_graph_"+key] = smart_umap.get_graph()
        
        ## Compute clustering
        print("Computing clusters")
        umap_graph = smart_umap.get_graph()
        smart_leiden = SmartLeiden(graph=umap_graph)

        if not isinstance(leiden_res, list):
            leiden_res = [leiden_res]

        for res in leiden_res:
            leiden_clusters = smart_leiden.cluster(resolution=res, partition_type='RBC')
            adata.obs["leiden_feature_" + key + "_" +str(res)] = leiden_clusters
            adata.obsm["leiden_feature_" + key + "_res_"+str(res)+"_one_hot"] = torch.nn.functional.one_hot(torch.from_numpy(leiden_clusters).long())  
        
        return adata


def conditional_motif_enrichment(adata: Anndata,
                                 feature_key: str,
                                 classify_or_regress: str,
                                 alpha_regularization: Union[float, list],
                                 verbose: bool = False):
    
    all_folds_test_anndatas = []

    if not isinstance(alpha_regularization, list):
        alpha_regularization = [alpha_regularization]
    
    if classify_or_regress == 'classify':
    
        ## Train classifier on spatially partitioned train/test folds
        for kfold in range(1,5):

            print(f"Running kfold {kfold}")
            
            ## spatial split
            train_anndata = adata[adata.obs[f'train_test_fold_{kfold}'] == 0]
            test_anndata = adata[adata.obs[f'train_test_fold_{kfold}'] == 1]

            ## balance after train/test split b/c of spatial partitioning to downsample majority class

            train_anndata = balance_anndata(train_anndata, 'classify_condition')
            
            ## Train classifier
            classifier = RidgeClassifierCV(alphas=alpha_regularization)
            classifier.fit(train_anndata.obsm[feature_key], train_anndata.obs['classify_condition'])

            if verbose:
                print("Fit alpha:")
                print(classifier.alpha_)

                ## Confirm train/test scores are similar to ensure regularization worked as expected
                print("Train score:")
                print(classifier.score(train_anndata.obsm[feature_key], train_anndata.obs['classify_condition']))

                print("Test score:")
                print(classifier.score(test_anndata.obsm[feature_key], test_anndata.obs['classify_condition']))
                
            ## Get conditional motif enrichment from trained classifier and write to anndata
            predicted_condition = classifier.decision_function(test_anndata.obsm[feature_key])
            test_anndata.obs['predicted_condition'] = predicted_condition
            all_folds_test_anndatas.append(test_anndata)
    elif classify_or_regress == "regress":
        print('to be implemented')


    ## Merge test folds together
    all_test_anndatas = merge_anndatas_inner_join(all_folds_test_anndatas)

    return all_test_anndatas

