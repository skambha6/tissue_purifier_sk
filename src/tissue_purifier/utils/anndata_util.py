from typing import List, Any
from anndata import Anndata

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