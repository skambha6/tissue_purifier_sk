from typing import Tuple, Union, List
import torch
import numpy
import pyro
import time
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.ops.special import get_quad_rule
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property
from pyro.infer import SVI, Trace_ELBO
import pandas as pd
import pyro.poutine
import pyro.optim
import matplotlib.pyplot as plt
from .gene_utils import GeneDataset

from pyro.distributions import Poisson

class GeneRegression:
    """
    Given the cell-type labels and some covariates the model predicts the gene expression.
    The counts are modelled as a Poisson distribution.
    """
    
    def __init__(self):
        self._optimizer = None
        self._optimizer_initial_state = None
        self._loss_history = []
        self._train_kargs = None
    
    ### TODO: double CHECK these methods ###
    
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
    
    ## double check this ##
    def _get_cell_type_names_kg(self, g) -> numpy.ndarray:
        """ Return a numpy.array of shape k_cell_type by g with the cell_type_names """
        inverse_cell_type_mapping = self._get_inverse_cell_type_mapping()
        k_cell_types = len(inverse_cell_type_mapping.keys())
        cell_types_codes = torch.arange(k_cell_types).view(-1, 1).expand(k_cell_types, g)
        cell_types_names_kg = numpy.array(list(inverse_cell_type_mapping.values()))[cell_types_codes.cpu().numpy()]
        return cell_types_names_kg

    ## double check this ##
    def _get_gene_names_kg(self, k: int) -> numpy.ndarray:
        """ Return a numpy.array of shape k by genes with the gene_names """
        gene_names_list = self._get_gene_list()
        len_genes = len(gene_names_list)
        gene_codes = torch.arange(len_genes).view(1, -1).expand(k, len_genes)
        gene_names_kg = numpy.array(gene_names_list)[gene_codes.cpu().numpy()]
        return gene_names_kg
    
    
    def _model(self,
           n_cells: int,
           l_cov: int,
           g_genes: int,
           k_cell_types: int,
           use_covariates: bool,
           counts_ng: torch.Tensor,
           total_umi_n: torch.Tensor,
           covariates_nl: torch.Tensor,
           cell_type_ids_n: torch.Tensor,
           subsample_size_cells: int,
           **kargs):
        
        
        # Define the right device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Define the plates (i.e. conditional independence). It make sense to subsample only cells.
        cell_plate = pyro.plate("cells", size=n_cells, dim=-1, device=device, subsample_size=subsample_size_cells)
        
        #cell_types_plate = pyro.plate("cell_types", size=k_cell_types, dim=-3, device=device) # don't need
        
        
        # Figure out a reasonable initialization for beta0_kg; initialize at zero for now
        beta0_k1g = pyro.param("beta0", torch.zeros((k_cell_types,1,g_genes)).to(device))
        
        if use_covariates:
            beta_klg = pyro.param("beta", torch.zeros((k_cell_types, l_cov, g_genes), device=device))
            covariate_nl1 = covariates_nl.unsqueeze(dim=-1).to(device)
        
        with cell_plate as ind_n:
            
            cell_ids_sub_n = cell_type_ids_n[ind_n].to(device)
            total_umi_sub_n1 = total_umi_n[ind_n, None].to(device)
            counts_sub_ng = counts_ng[ind_n]
            
            beta0_sub_n1g = beta0_k1g[cell_ids_sub_n]
            
            if use_covariates:
                covariate_sub_nl1 = covariate_nl1[ind_n]
                
                beta_sub_nlg = beta_klg[cell_ids_sub_n]
                
                log_mu_sub_n1g = beta0_sub_n1g + torch.sum(covariate_sub_nl1 * beta_sub_nlg, dim=-2, keepdim=True)
                log_mu_sub_ng = log_mu_sub_n1g.squeeze()
                
            else:
                
                assert beta0_sub_n1g.shape[-3] != 1
                assert beta0_sub_n1g.shape[-1] != 1
                
                log_mu_sub_ng = beta0_sub_n1g.squeeze() ##assume n / g aren't 1
                
            return pyro.sample("counts",
                Poisson(rate=total_umi_sub_n1*(log_mu_sub_ng.exp())).to_event(1),
                obs=counts_sub_ng.to(device))


        
    def _guide(self,            
           n_cells: int,
           l_cov: int,
           g_genes: int,
           k_cell_types: int,
           use_covariates: bool,
           counts_ng: torch.Tensor,
           total_umi_n: torch.Tensor,
           covariates_nl: torch.Tensor,
           cell_type_ids_n: torch.Tensor,
           subsample_size_cells: int,
           **kargs):
        
        pass
    
    @property
    def optimizer(self) -> pyro.optim.PyroOptim:
        """ The optimizer associated with this model. """
        assert self._optimizer is not None, "Optimizer is not specified. Call configure_optimizer first."
        return self._optimizer
    
    def configure_optimizer(self,
                            optimizer_type: str = 'adam',
                            lr: float = 5E-3,
                            betas: Tuple[float, float] = (0.9, 0.999),
                            momentum: float = 0.9,
                            alpha: float = 0.99):
        """
        Configure the optimizer to use.

        Args:
            optimizer_type: Either 'adam' (default), 'sgd' or 'rmsprop'
            lr: learning rate
            betas: betas for 'adam' optimizer. Ignored if :attr:`optimizer_type` is not 'adam'.
            momentum: momentum for 'sgd' optimizer. Ignored if :attr:`optimizer_type` is not 'sgd'.
            alpha: alpha for 'rmsprop' optimizer. Ignored if :attr:`optimizer_type` is not 'rmsprop'.
        """
        if optimizer_type == 'adam':
            self._optimizer = pyro.optim.Adam({"lr": lr, "betas": betas})
        elif optimizer_type == 'sgd':
            self._optimizer = pyro.optim.SGD({"lr": lr, "momentum": momentum})
        elif optimizer_type == 'rmsprop':
            self._optimizer = pyro.optim.RMSprop({"lr": lr, "alpha": alpha})
        else:
            raise ValueError("optimizer_type should be either 'adam', 'sgd' or 'rmsprop'. \
                              Received {}".format(optimizer_type))
        self._optimizer_initial_state = self._optimizer.get_state()
        
        
    def show_loss(self, figsize: Tuple[float, float] = (4, 4), logx: bool = False, logy: bool = False, ax=None):
        """
        Show the loss history. Useful for checking if the training has converged.

        Args:
            figsize: the size of the image. Used only if ax=None
            logx: if True the x_axis is shown in logarithmic scale
            logy: if True the x_axis is shown in logarithmic scale
            ax: The axes object to draw the plot onto. If None (defaults) creates a new figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self._loss_history)

        if logx:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")

        if logy:
            ax.set_yscale("log")
        else:
            ax.set_xscale("linear")

        if ax is None:
            fig.tight_layout()
            plt.close(fig)
            return fig
        
    def train(self,
              dataset: GeneDataset,
              n_steps: int = 2500,
              use_covariates: bool = True,
              print_frequency: int = 50,
              subsample_size_cells: int = None,
              **kargs
              ):
        """
        Train the model. The trained parameter are stored in the pyro.param_store and
        can be accessed via :meth:`get_params`.

        Args:
            dataset: Dataset to train the model on
            n_steps: number of training step
            print_frequency: how frequently to print loss to screen
            subsample_size_genes: for large dataset, the minibatch can be created using a subset of genes.
            kargs: unused parameters

        Note:
            If you get an out-of-memory error try to tune the :attr:`subsample_size_cells`
            and :attr:`subsample_size_genes`.
        """
        
        # prepare train kargs dict
        train_kargs = {
            'use_covariates': use_covariates,
            'subsample_size_cells': subsample_size_cells,
            'cell_type_mapping': dataset.cell_type_mapping,
            'gene_names': dataset.gene_names,
        }
        
        # make a copy so that can edit train_kargs without changing _train_kargs
        self._train_kargs = train_kargs.copy()
        
        # Unpack the dataset
        counts_ng = dataset.counts.long()
        cell_type_ids = dataset.cell_type_ids.long()
        total_umi_n = counts_ng.sum(dim=-1)
        
        # Prepare arguments for training
        train_kargs["n_cells"] = counts_ng.shape[0]
        train_kargs["g_genes"] = counts_ng.shape[1]
        train_kargs["l_cov"] = dataset.covariates.shape[-1]
        train_kargs["k_cell_types"] = dataset.k_cell_types
        train_kargs["counts_ng"] = counts_ng.cpu()
        train_kargs["total_umi_n"] = total_umi_n.cpu()
        train_kargs["cell_type_ids_n"] = cell_type_ids
        train_kargs["covariates_nl"] = dataset.covariates.float().cpu()
        
        beta0_k1g_store = []
        beta_klg_store = []
        
        start_time = time.time()
        svi = SVI(self._model, self._guide, self.optimizer, loss=Trace_ELBO())
        for i in range(n_steps + 1):
            loss = svi.step(**train_kargs)
            self._loss_history.append(loss)
            if (i % print_frequency == 0):
                print('[iter {}]  loss: {:.4f}'.format(i, loss))
            
            beta0_k1g = pyro.get_param_store().get_param("beta0").float().cpu()
            beta0_k1g_store.append(beta0_k1g)
            
            if use_covariates:
                beta_klg = pyro.get_param_store().get_param("beta").float().cpu()
                beta_klg_store.append(beta_klg)
            
        print("Training completed in {} seconds".format(time.time()-start_time))
        
        return beta0_k1g_store, beta_klg_store
    
    
    @torch.no_grad()
    def predict(self,
                dataset: GeneDataset,
                num_samples: int = 10,
                subsample_size_cells: int = None,
                use_covariates: bool = False) -> (torch.tensor):
        
        
        n, g = dataset.counts.shape[:2]
        k = dataset.k_cell_types

        # params
        beta0_k1g = pyro.get_param_store().get_param("beta0").float().cpu()

        # dataset
        counts_ng = dataset.counts.long().cpu()
        cell_type_ids = dataset.cell_type_ids.long().cpu()
        
        
    
        # prepare storage
        device_calculation = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pred_counts_ng = torch.zeros((n, g), dtype=torch.long, device=torch.device("cpu"))
        q_ng = torch.zeros((n, g), dtype=torch.float, device=torch.device("cpu"))
        
        # Loop to fill the predictions for all cell and genes
        subsample_size_cells = n if subsample_size_cells is None else subsample_size_cells
        
        for n_left in range(0, n, subsample_size_cells):
            n_right = min(n_left + subsample_size_cells, n)

            subn_cell_ids = cell_type_ids[n_left:n_right]
            subn_counts_ng = counts_ng[n_left:n_right]
            subn_total_umi_n1 = subn_counts_ng.sum(dim=-1, keepdim=True)

            beta0_n1g = beta0_k1g[subn_cell_ids]
 
            if use_covariates:
                n, l = dataset.covariates.shape[:2]
                covariates_nl1 = dataset.covariates.unsqueeze(dim=-1).float().cpu()
                beta_klg = pyro.get_param_store().get_param("beta").float().cpu()
                beta_nlg = beta_klg[subn_cell_ids]
                
                subn_covariates_nl1 = covariates_nl1[n_left:n_right]
                log_rate_n1g = beta0_n1g + torch.sum(subn_covariates_nl1 *
                                                     beta_nlg,
                                                     dim=-2, keepdim=True)
                # log_rate_n1g = torch.sum(subn_covariates_nl1 *
                #                                      beta_nlg,
                #                                      dim=-2, keepdim=True)
            else:
                log_rate_n1g = beta0_n1g

            rate = subn_total_umi_n1.to(device_calculation) * log_rate_n1g.squeeze(dim=-2).exp().to(device_calculation)
            mydist = Poisson(rate=rate)

            pred_counts_tmp_bng = mydist.sample(sample_shape=torch.Size([num_samples])).cpu()

            ## TODO double check the mean statement here
            pred_counts_ng[n_left:n_right] = pred_counts_tmp_bng.mean(dim=-3).long().cpu()
            #pred_counts_ng[n_left:n_right] = pred_counts_tmp_bng[0].long().cpu()
            
            
             ## TODO double check the subtract/mean statement here
            q_ng_tmp = (pred_counts_tmp_bng - subn_counts_ng).abs().float().mean(dim=-3)
            q_ng[n_left:n_right] = q_ng_tmp.cpu()


       # average by cell_type to obtain q_prediction
        unique_cell_types = torch.unique(cell_type_ids)
        q_kg = torch.zeros((unique_cell_types.shape[0], g), dtype=torch.float, device=torch.device("cpu"))
        for k, cell_type in enumerate(unique_cell_types):
            mask = (cell_type_ids == cell_type)
            q_kg[k] = q_ng[mask].mean(dim=0)

        # Compute df_metric_kg
        # combine: gene_names_kg, cell_types_names_kg, q_kg, q_data_kg, log_score_kg into a dataframe
        cell_types_names_kg = self._get_cell_type_names_kg(g=len(dataset.gene_names))
        k_cell_types, len_genes = cell_types_names_kg.shape
        gene_names_kg = self._get_gene_names_kg(k=k_cell_types)
        assert gene_names_kg.shape == cell_types_names_kg.shape == q_kg.shape, \
            "Shape mismatch {0} vs {1} vs {2} vs {3}".format(gene_names_kg.shape,
                                                             cell_types_names_kg.shape,
                                                             q_kg.shape)

        df_metric_kg = pd.DataFrame(cell_types_names_kg.flatten(), columns=["cell_type"])
        df_metric_kg["gene"] = gene_names_kg.flatten()
        df_metric_kg["q_dist"] = q_kg.flatten().cpu().numpy()

        # df_counts_ng = pd.DataFrame(pred_counts_ng.flatten().cpu().numpy(), columns=["counts_pred"])
        # df_counts_ng["counts_obs"] = dataset.counts.flatten().cpu().numpy()
        # df_counts_ng["cell_type"] = cell_names_ng.flatten()
        # df_counts_ng["gene"] = gene_names_ng.flatten()

        return df_metric_kg, pred_counts_ng