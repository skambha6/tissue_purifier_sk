from typing import List, Any
import anndata as Anndata
import numpy as np

# Set of simple helper functions to manipulate anndatas

def merge_anndatas_inner_join(anndata_list):
    """
    Merge a list of anndata objects using an inner join operation.

    Args:
        anndata_list (list): List of anndata objects to be merged.

    Returns:
        anndata.AnnData: The merged anndata object.
    """
    # Check if the anndata_list is not empty
    if not anndata_list:
        raise ValueError("Input anndata_list is empty.")

    # Determine the common feature names across all anndata objects
    common_feature_names = anndata_list[0].var_names
    for ad in anndata_list[1:]:
        common_feature_names = common_feature_names.intersection(ad.var_names)

    # Filter observations for each anndata object using the common feature names
    filtered_anndata_list = []
    for ad in anndata_list:
        ad_filtered = ad[:, common_feature_names]
        filtered_anndata_list.append(ad_filtered)

    # Concatenate the list of filtered anndata objects along axis 0 (rows)
    merged_anndata = Anndata.concat(filtered_anndata_list, axis=0)

    return merged_anndata

def balance_anndata(adata, label_key):
    """
    Balances the classes in an anndata object stored in obs under 'label_key' by downsampling the majority class.

    Args:
        adata: the anndata object to be balanced
        label_key: the key in adata.obs where the class labels are stored

    Returns:
        anndata.AnnData: The balanced anndata object.
    """

    # Assuming you have an AnnData object `adata` with the class label column in adata.obs['label']

    # Calculate class counts
    class_counts = adata.obs[label_key].value_counts()

    # Identify majority and minority classes
    majority_class_label = class_counts.idxmax()
    minority_class_label = class_counts.idxmin()

    # Separate majority and minority classes
    adata_majority = adata[adata.obs[label_key] == majority_class_label].copy()
    adata_minority = adata[adata.obs[label_key] == minority_class_label].copy()

    # Downsample majority class
    np.random.seed(123)  # for reproducibility
    random_subset_indices = np.random.choice(adata_majority.shape[0], size=len(adata_minority), replace=False)
    adata_majority_downsampled = adata_majority[random_subset_indices]

    # Combine minority class with downsampled majority class
    adata_balanced = adata_minority.concatenate(adata_majority_downsampled)
    
    return adata_balanced