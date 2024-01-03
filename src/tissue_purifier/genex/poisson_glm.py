from typing import Tuple, Union, List
import torch
import numpy
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from tissue_purifier.genex.gene_utils import GeneDataset # relat0ive vs absolute imports 


from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import d2_tweedie_score, make_scorer
d2_poisson_score = make_scorer(d2_tweedie_score, power=1)

from sklearn._loss.loss import HalfPoissonLoss


class GeneRegression:
    """
    Given the cell-type labels and some covariates the model predicts the gene expression.
    The counts are modelled as a Poisson process. See documentation for more details.
    """
    
    def __init__(self, use_covariates, umi_scaling, cell_type_prop_scaling, scale_covariates: bool = False):
        self._train_kargs = None
        self.clf_g = {}
        self._use_covariates = use_covariates
        self._scale_covariates = scale_covariates
        self._alpha_dict = {}
        self._umi_scaling = umi_scaling
        self._cell_type_prop_scaling = cell_type_prop_scaling
        
        self._scaler = StandardScaler() # MinMaxScaler()
        
        ## assert umi scaling and cell type prop scaling are integers > 0
        
    def _get_regression_X(self, dataset) -> np.array:
        
        counts_ng = dataset.counts.long()
        cell_type_ids = dataset.cell_type_ids.long()
        total_umi_n = counts_ng.sum(dim=-1)
        covariates_nl = dataset.covariates.clone().float().cpu()
        cell_type_props_nk = self._cell_type_prop_scaling*dataset.cell_type_props
        
        log_total_umi_n1 = self._umi_scaling*np.log(total_umi_n+1).reshape(-1,1)
        
        if self._scale_covariates:
            # Check if the scaler has been fit
            # if scaler has been fit, means training of GeneRegression has completed
            if hasattr(self._scaler, 'scale_'):
                print("scaling covariates")
                covariates_nl = self._scaler.transform(covariates_nl)
            else:
                print("fitting scaler and scaling covariates")
                covariates_nl = self._scaler.fit_transform(covariates_nl)
                
        if self._use_covariates:
            X = np.concatenate([log_total_umi_n1, 
                                cell_type_props_nk,
                                covariates_nl], axis=1) 
        else:
             X = np.concatenate([log_total_umi_n1, 
                            cell_type_props_nk], axis=1)
            
        return X
    
    def _get_gene_list(self) -> List[str]:
        return self._train_kargs["gene_names"]

    def _get_cell_type_mapping(self) -> dict:
        return self._train_kargs["cell_type_mapping"]

    def _get_inverse_cell_type_mapping(self) -> dict:
        cell_type_mapping = self._get_cell_type_mapping()

        # Invert the cell_type_mapping (of the form: "cell_type" -> integer code)
        # to inverse_cell_type_mapping (of the form: integer_code -> "cell_types")
        # Note that multiple cell_types can be assigned to the same integer codes thefefor the inversion need
        # to keep track of possible name collisions
        inverse_cell_type_mapping = dict()
        for cell_type_name, code in cell_type_mapping.items():
            try:
                existing = inverse_cell_type_mapping[code]
                inverse_cell_type_mapping[code] = existing + "_AND_" + str(cell_type_name)
            except KeyError:
                inverse_cell_type_mapping[code] = str(cell_type_name)
        return inverse_cell_type_mapping

    def _get_cell_type_names_kg(self, g) -> numpy.ndarray:
        """ Return a numpy.array of shape k_cell_type by g with the cell_type_names """
        inverse_cell_type_mapping = self._get_inverse_cell_type_mapping()
        k_cell_types = len(inverse_cell_type_mapping.keys())
        cell_types_codes = torch.arange(k_cell_types).view(-1, 1).expand(k_cell_types, g)
        cell_types_names_kg = numpy.array(list(inverse_cell_type_mapping.values()))[cell_types_codes.cpu().numpy()]
        return cell_types_names_kg

    def _get_gene_names_kg(self, k: int) -> numpy.ndarray:
        """ Return a numpy.array of shape k by genes with the gene_names """
        gene_names_list = self._get_gene_list()
        len_genes = len(gene_names_list)
        gene_codes = torch.arange(len_genes).view(1, -1).expand(k, len_genes)
        gene_names_kg = numpy.array(gene_names_list)[gene_codes.cpu().numpy()]
        return gene_names_kg
    
    def get_alpha_dict(self) -> dict:
        """ Return alpha regularization dictionary with per_gene alpha """
        return self._alpha_dict
    
    def set_alpha_dict(self, alpha_dict: dict):
        """ Set alpha regularization dictionary with per_gene alpha """
        self._alpha_dict = alpha_dict
    
    def train(self,
              train_dataset: GeneDataset,
              val_dataset: GeneDataset = None,
              n_steps: int = 100,
              print_frequency: int = 100,
              regularization_sweep: bool = False,
              fit_intercept: bool = False,
              alpha_regularization_strengths: np.array = np.array([0.01]),
              alpha_dict: dict = None,
              metric: str = None, ## return dsquared by default 
              **kargs
              ):
        
        ## Create fit_alpha_dict flag based on whether alpha_dict is passed. If alpha_dict is passed, set fit_alpha_dict flag to False
        ## If fit_alpha_dict flag is false, fit_model() will not over-write alpha_dict
        fit_alpha_dict = True
        if alpha_dict is not None:
            print("Using user-supplied alpha regularization dictionary")
            fit_alpha_dict = False
            
            
        # prepare train kargs dict
        train_kargs = {
            'n_steps': n_steps,
            'regularization_sweep': regularization_sweep,
            'alpha_regularization_strengths': alpha_regularization_strengths,
            'cell_type_mapping': train_dataset.cell_type_mapping,
            'gene_names': train_dataset.gene_names,
            'fit_alpha_dict': fit_alpha_dict
        }
        
        # make a copy so that can edit train_kargs without changing _train_kargs
        self._train_kargs = train_kargs.copy()

        # Unpack the dataset
        counts_ng_train = train_dataset.counts.long()
        cell_type_ids = train_dataset.cell_type_ids.long()
        total_umi_n = counts_ng_train.sum(dim=-1)
        
        # Prepare validation counts if regularization_sweep is true
        if regularization_sweep:
            assert val_dataset is not None
            counts_ng_val = val_dataset.counts.long()
        else:
            counts_ng_val = None
        
        # Prepare arguments for training
        train_kargs["n_cells"] = counts_ng_train.shape[0]
        train_kargs["g_genes"] = counts_ng_train.shape[1]
        train_kargs["l_cov"] = train_dataset.covariates.shape[-1]
        train_kargs["k_cell_types"] = train_dataset.k_cell_types
        train_kargs["counts_ng"] = counts_ng_train.cpu()
        train_kargs["total_umi_n"] = total_umi_n.cpu()
        train_kargs["covariates_nl"] = train_dataset.covariates.float().cpu()
        train_kargs["cell_type_ids_n"] = cell_type_ids
        train_kargs["cell_type_props"] = train_dataset.cell_type_props 
        train_kargs["alpha_dict"] = alpha_dict
        
        self._train_kargs = train_kargs
        
        # fit model
        self.fit_model(train_dataset, val_dataset, counts_ng_train, counts_ng_val, fit_intercept, print_frequency, train_kargs)

    def fit_model(self, train_dataset: GeneDataset, 
                        val_dataset: GeneDataset,
                        counts_ng_train: np.array,
                        counts_ng_val: np.array,
                        fit_intercept: bool,
                        print_frequency: int,
                        kargs: dict):
        
        ## prepare coefficients for regression
        total_umi_n = kargs["total_umi_n"]
        cell_type_props_nk = kargs['cell_type_props']
        covariates_nl = kargs['covariates_nl']
        
        n_steps = kargs['n_steps']
        g_genes = kargs['g_genes']
        gene_names = kargs['gene_names']
        regularization_sweep = kargs['regularization_sweep']
        alpha_regularization_strengths = kargs['alpha_regularization_strengths']
        fit_alpha_dict = kargs["fit_alpha_dict"]
        alpha_dict = kargs["alpha_dict"]
        
        ## get coefficients for regression
        
        X_train = self._get_regression_X(train_dataset)
        
        if regularization_sweep:
            X_val = self._get_regression_X(val_dataset)
    
        if not regularization_sweep:
            if fit_alpha_dict:
                 ## TODO: assert that alpha regularization is a numpy array
                assert alpha_regularization_strengths.size == 1
                for gene_name in gene_names:
                    self._alpha_dict[gene_name] = alpha_regularization_strengths[0]
            else:
                assert alpha_dict is not None
                ## TODO: assert all genes are present in user given alpha_dict
                self._alpha_dict = alpha_dict
        
        ## Train a separate GLM for each gene
    
        print('start fitting genes')
        for g_ind in range(g_genes):
            
            
            import time
            
            ## output progress message
            if (g_ind+1) % print_frequency == 0:
                print('finished ' + str(g_ind+1) + ' genes')
                
            # get gene name
            gene_name = gene_names[g_ind]
            
            # get gene counts for fitting model
            y_train = np.ravel(counts_ng_train[:,g_ind])
            
            # get gene counts for choosing alpha regularization value
            if regularization_sweep:
                y_val = np.ravel(counts_ng_val[:,g_ind])
                val_scores = self.regularization_validation(X_train, y_train, X_val, y_val, alpha_regularization_strengths, n_steps)
                ## choose alpha value based on validation set performance
                self._alpha_dict[gene_name] = alpha_regularization_strengths[np.argmax(val_scores)]
                
            # import pdb; pdb.set_trace()
            
            # performance profiling
            start_time = time.time()
            ## fit GLM 
            self.clf_g[gene_name] = PoissonRegressor(alpha = self._alpha_dict[gene_name], fit_intercept=fit_intercept, solver='newton-cholesky', max_iter=n_steps).fit(X_train, y_train)
            end_time = time.time()
            # print(str(end_time-start_time) + ' seconds elapsed per gene')
            
                
        ## output median alpha if regularization sweep was run
        if regularization_sweep:
            print("median alpha: " + str(np.median(list(self._alpha_dict.values()))))
    
    
    def regularization_validation(self, X_train, y_train, X_val, y_val, alphas, n_steps):
        
        ## Train separate GLM for each alpha and output corresponding d_sq scores on validation set
        val_scores = []

        for alpha in alphas:
            estimator = PoissonRegressor(alpha=alpha, fit_intercept=False, solver='newton-cholesky', max_iter=n_steps)
            estimator.fit(X_train,y_train)
            val_score = estimator.score(X_val, y_val)
            
            val_scores.append(val_score)

        return val_scores
            

    def predict(self,
            dataset: GeneDataset,
            return_true_counts: bool = False) -> (pd.DataFrame, pd.DataFrame):
        
        # constants
        n, g = dataset.counts.shape[:2]
        n, l = dataset.covariates.shape[:2]
        k = dataset.k_cell_types

        # dataset
        counts_ng = np.array(dataset.counts.long().cpu())
        cell_type_ids = dataset.cell_type_ids.long().cpu()
        covariates_nl1 = dataset.covariates.unsqueeze(dim=-1).float().cpu()

        pred_counts_ng = -1*np.ones((n, g))
        print("predicting")

        X = self._get_regression_X(dataset)
        gene_list = self._get_gene_list()
        for g_ind in range(g):
            pred_counts_n1 = self.clf_g[gene_list[g_ind]].predict(X)
            pred_counts_ng[:,g_ind] = pred_counts_n1
            
        if return_true_counts:
            return pred_counts_ng, counts_ng
        else:
            return pred_counts_ng 
            
    @staticmethod
    def compute_eval_metrics(pred_counts_ng: np.array,
                            counts_ng: np.array,
                            cell_type_ids: np.array,
                            gene_names: np.array,
                            pred_counts_ng_baseline: np.array = None,
                            baseline_sample_size: int = 10000) -> (pd.DataFrame, pd.DataFrame):
        
        n, g = counts_ng.shape[:2]
        unique_cell_types = np.unique(cell_type_ids)
        k = len(unique_cell_types)
        
        import sklearn
        
        ## compute d_sq, stratified by cell type 
        d_sq_kg = np.zeros((unique_cell_types.shape[0],g))
        for k, cell_type in enumerate(unique_cell_types):
            mask = (cell_type_ids == cell_type)
            for g_ind in range(g):
                d_sq_kg[k,g_ind] = GeneRegression.compute_d2(y_true=counts_ng[mask,g_ind], y_pred=pred_counts_ng[mask,g_ind])
            
        # print("reached")
        ## TODO: optionally compute d_sq across all cell types?
            
        ## compute q_dist
        q_ng = np.absolute(pred_counts_ng - counts_ng)
        
        # average qdist by cell_type to obtain q_prediction
        q_kg = np.zeros((unique_cell_types.shape[0], g))
        
        for k, cell_type in enumerate(unique_cell_types):
            mask = (cell_type_ids == cell_type)
            q_kg[k] = np.mean(q_ng[mask], axis=0)
            
        ## convert to dataframes
        d_sq_gk = np.transpose(d_sq_kg)
        df_d_sq_gk = pd.DataFrame(d_sq_gk, columns=unique_cell_types)
        df_d_sq_gk.index = gene_names
        
        q_gk = np.transpose(q_kg)
        df_q_gk = pd.DataFrame(q_gk, columns=unique_cell_types)
        df_q_gk.index = gene_names
        
        ## if baseline predicted counts given, z-score q_dist metric
        if pred_counts_ng_baseline is not None:

            # d_sq_g_baseline = compute_d2(pred_counts_ng_baseline, counts_ng)
            q_ng_baseline = np.absolute(pred_counts_ng_baseline - counts_ng)
            
            q_baseline_mu_kg = np.zeros((unique_cell_types.shape[0], g))
            q_baseline_std_kg = np.zeros((unique_cell_types.shape[0], g))
            
            for k, cell_type in enumerate(unique_cell_types):
                mask = (cell_type_ids == cell_type)
                q_tg_baseline = q_ng_baseline[mask]

                sample_ind = np.random.choice(q_tg_baseline.shape[0], size=baseline_sample_size)
                q_tg_baseline_sample = q_tg_baseline[sample_ind]

                q_baseline_mu_kg[k] = q_tg_baseline_sample.mean(axis=0)
                q_baseline_std_kg[k] = q_tg_baseline_sample.std(axis=0)
                
                
            q_z_kg = (q_kg - q_baseline_mu_kg)/q_baseline_std_kg
            
            q_z_gk = np.transpose(q_z_kg)
            df_q_z_gk = pd.DataFrame(q_z_gk, columns=unique_cell_types)
            df_q_z_gk.index = gene_names
            
            return df_d_sq_gk, df_q_z_gk
        else:
            return df_d_sq_gk, df_q_gk
        
    from sklearn._loss.loss import HalfPoissonLoss
        
    ## implementation from sklearn: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_glm/glm.py#L464
    @staticmethod
    def compute_d2(y_pred, y_true):

        # self._base_loss.link.inverse(raw_prediction)

        y = y_true.astype('float64')
        base_loss = HalfPoissonLoss()
        
        if not base_loss.in_y_true_range(y):
            raise ValueError(
                "Some value(s) of y are out of the valid range of the loss"
                f" {base_loss.__name__}."
            )
            
        if not base_loss.in_y_true_range(y_pred):
            raise ValueError(
                "Some value(s) of y pred are out of the valid range of the loss"
                f" {base_loss.__name__}."
            )

        raw_prediction = base_loss.link.link(y_pred).astype('float64')


        constant = np.average(
            base_loss.constant_to_optimal_zero(y_true=y, sample_weight=None),
            weights=None,
        )

        # Missing factor of 2 in deviance cancels out.
        deviance = base_loss(
            y_true=y,
            raw_prediction=raw_prediction
        )
        y_mean = base_loss.link.link(np.average(y, weights=None))
        deviance_null = base_loss(
            y_true=y,
            raw_prediction=np.tile(y_mean, y.shape[0]),
            sample_weight=None,
            n_threads=1,
        )
        return 1 - (deviance + constant) / (deviance_null + constant)

        
        
    
            
        
        

        
        
        