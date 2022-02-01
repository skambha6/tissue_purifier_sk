from typing import Sequence, List, Any, Dict

import torch
from argparse import ArgumentParser
from torch.nn import functional as F
import numpy
import math

from tissue_purifier.model_utils.resnet_backbone import make_resnet_backbone
from tissue_purifier.model_utils.benckmark_mixin import BenchmarkModelMixin
from tissue_purifier.misc_utils.misc import LARS
from tissue_purifier.misc_utils.misc import (
    smart_bool,
    linear_warmup_and_cosine_protocol)


def dino_loss(output_t: torch.Tensor,
              output_s: torch.Tensor,
              ncrops_t: int,
              ncrops_s: int,
              temp_t: torch.Tensor,
              temp_s: torch.Tensor,
              center_t: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Returns:
        total_loss, prob_t, prob_s
    """
    all_logit_s = output_s / temp_s
    all_prob_s = F.softmax(all_logit_s, dim=-1)
    all_log_prob_s = F.log_softmax(all_logit_s, dim=-1)
    prob_s = all_prob_s.chunk(ncrops_s)
    log_prob_s = all_log_prob_s.chunk(ncrops_s)

    all_logit_t = (output_t - center_t).detach() / temp_t
    all_prob_t = F.softmax(all_logit_t, dim=-1)
    all_log_prob_t = F.log_softmax(all_logit_t, dim=-1)
    prob_t = all_prob_t.chunk(ncrops_t)
    log_prob_t = all_log_prob_t.chunk(ncrops_t)

    total_loss = 0
    n_loss_terms = 0
    for iq, (q, log_q) in enumerate(zip(prob_t, log_prob_t)):
        for ip, (p, log_p) in enumerate(zip(prob_s, log_prob_s)):
            if ip == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            # TODO: There is inconsistency in paper vs code. SYmmetric or non-symmetric version of KL
            # loss = -0.5 * (p * log_q + q * log_p).sum(dim=-1)  # shape: BATCH_SIZE
            loss = -(q * log_p).sum(dim=-1)  # shape: BATCH_SIZE
            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms
    return total_loss, all_prob_t, all_prob_s


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("WARNINGS: The mean is more than 2 std from [a, b] in nn.init.trunc_normal_. \
              The distribution of values may be incorrect.")

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DINOHead(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_bn=False, norm_last_layer=True):
        super().__init__()
        assert len(hidden_dim) >= 1
        sizes = [in_dim] + hidden_dim + [out_dim]

        layers = []
        for i in range(len(sizes) - 2):
            if use_bn:
                layers.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(torch.nn.BatchNorm1d(sizes[i + 1]))
            else:
                layers.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=True))
            layers.append(torch.nn.GELU())
        self.mlp = torch.nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = torch.nn.utils.weight_norm(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiResolutionNet(torch.nn.Module):
    """
    Net that can act on both a single torch.Tensor or a list of torch.Tensors
    with possibly different spatiual resolutions.
    """
    def __init__(self,
                 backbone_type: str,
                 backbone_in_ch: int,
                 head_hidden_chs: List[int],
                 head_out_ch: int,
                 head_use_bn: bool):
        super().__init__()

        self.backbone = make_resnet_backbone(
            backbone_in_ch=backbone_in_ch,
            backbone_type=backbone_type)
        in_tmp = torch.zeros((1, self.backbone_in_ch, 32, 32))
        out_tmp = self.backbone(in_tmp)
        head_ch_in = out_tmp.shape[1]

        self.head = DINOHead(
            in_dim=head_ch_in,
            hidden_dim=head_hidden_chs,
            out_dim=head_out_ch,
            use_bn=head_use_bn,
            norm_last_layer=True)

    @staticmethod
    def init_projection(
            ch_in: int,
            ch_out: int,
            ch_hidden: List[int]=None):

        sizes = [ch_in] + ch_hidden + [ch_out]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(torch.nn.BatchNorm1d(sizes[i + 1]))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """ x is either a torch.Tensor or a list of torch.Tensor of possibly different resolutions.
            1. concatenate tensor with same resolution and run the backbone
            2. concatenate the features
            3. run the head just one on the concatenated features
        """
        # convert to list
        if not isinstance(x, list):
            output = self.backbone(x)  # output is of size (b, c)
        else:
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True)[1], dim=0)

            start_idx, output = 0, None
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx], dim=0))

                # accumulate outputs
                if output is None:
                    output = _out
                else:
                    output = torch.cat((output, _out), dim=0)

                start_idx = end_idx

        # Run the head forward on the concatenated features.
        return self.head(output), output


class DinoModel(BenchmarkModelMixin):
    """
    See
    https://pytorch-lightning.readthedocs.io/en/stable/starter/style_guide.html  and
    https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L61-L301

    DINO implementation, inspired by:
    https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/2021/08/01/dino-self-supervised-vision-transformers.html
    And original github repo
    """
    def __init__(
            self,
            # architecture
            image_in_ch: int,
            backbone_type: str,
            head_hidden_chs: List[int],
            head_use_bn: bool,
            head_out_ch: int,
            # teacher centering
            center_momentum: float,
            # optimizer
            optimizer_type: str,
            # scheduler
            warm_up_epochs: int,
            warm_down_epochs: int,
            max_epochs: int,
            min_learning_rate: float,
            max_learning_rate: float,
            min_weight_decay: float,
            max_weight_decay: float,
            # validation
            val_iomin_threshold: float = 0.0,
            # temperatures
            set_temperature_using_ipr_init: bool = False,
            ipr_teacher_init: float = 40.0,
            ipr_student_init: float = 80.0,
            temperature_teacher_init: float = 0.04,
            temperature_student_init: float = 0.1,
            # the teacher's parameters update slowly and then stop
            param_momentum_init: float = 0.996,
            param_momentum_final: float = 0.996,
            param_momentum_epochs_end: int = 1000,
            **kwargs,

            ):
        super(DinoModel, self).__init__(val_iomin_threshold=val_iomin_threshold)

        # Next two lines will make checkpointing much simpler. Always keep them as-is
        self.save_hyperparameters()  # all hyperparameters are saved to the checkpoint
        self.neptune_run_id = None  # if from scratch neptune_experiment_is is None

        # architecture
        self.student = MultiResolutionNet(
            backbone_type=backbone_type,
            backbone_in_ch=image_in_ch,
            head_hidden_chs=head_hidden_chs,
            head_out_ch=head_out_ch,
            head_use_bn=head_use_bn,
        )

        # this is creating a separate teacher object with the same weights as the student object
        self.teacher = MultiResolutionNet(
            backbone_type=backbone_type,
            backbone_in_ch=image_in_ch,
            head_hidden_chs=head_hidden_chs,
            head_out_ch=head_out_ch,
            head_use_bn=head_use_bn,
        )
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.dim_out = self.teacher.head_out_ch
        self.register_buffer("center_teacher", torch.zeros(self.dim_out, requires_grad=False).float())
        self.register_buffer("population_t", torch.zeros(self.dim_out, requires_grad=False).float())
        self.register_buffer("population_s", torch.zeros(self.dim_out, requires_grad=False).float())

        # teacher parameters
        self.set_temperature_using_ipr_init = set_temperature_using_ipr_init
        if self.set_temperature_using_ipr_init:
            self.ipr_teacher_init = float(ipr_teacher_init)
            self.ipr_student_init = float(ipr_student_init)
            self.register_buffer("student_temperature",
                                 float(1.0) * torch.ones(1, requires_grad=False).float())
            self.register_buffer("teacher_temperature",
                                 float(1.0) * torch.ones(1, requires_grad=False).float())
        else:
            self.ipr_teacher_init = -1.0
            self.ipr_student_init = -1.0
            self.register_buffer("student_temperature",
                                 float(temperature_student_init) * torch.ones(1, requires_grad=False).float())
            self.register_buffer("teacher_temperature",
                                 float(temperature_teacher_init) * torch.ones(1, requires_grad=False).float())

        self.teacher_center_momentum = float(center_momentum)
        self.teacher_parameter_momentum_fn = linear_warmup_and_cosine_protocol(
            f_values=(param_momentum_init, param_momentum_init, param_momentum_final),
            x_milestones=(0, 0, 0, param_momentum_epochs_end))

        # optimizer
        self.optimizer_type = optimizer_type

        # scheduler
        assert warm_up_epochs + warm_down_epochs <= max_epochs
        self.learning_rate_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_learning_rate, max_learning_rate, min_learning_rate),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))
        self.weight_decay_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_weight_decay, min_weight_decay, max_weight_decay),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))

    @classmethod
    def add_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')

        # validation
        parser.add_argument("--val_iomin_threshold", type=float, default=0.0,
                            help="during validation, only patches with IoMinArea < IoMin_threshold are used "
                                 "in the kn-classifier and kn-regressor.")

        # architecture
        parser.add_argument("--image_in_ch", type=int, default=3, help="number of channels in the input images")
        parser.add_argument("--backbone_type", type=str, default="resnet34", help="backbone type",
                            choices=['resnet18', 'resnet34', 'resnet50'])
        parser.add_argument("--head_hidden_chs", type=int, nargs='+', default=[256, 512], help="head hidden channels")
        parser.add_argument("--head_out_ch", type=int, default=512, help="head output channels")
        parser.add_argument("--head_use_bn", type=smart_bool, default=True,
                            help="use batch normalization layers in the DINOHead")

        # optimizer
        parser.add_argument("--optimizer_type", type=str, default='adam', help="optimizer type",
                            choices=['adamw', 'lars', 'sgd', 'adam', 'rmsprop'])

        # Updating of the teacher network
        parser.add_argument("--param_momentum_init", type=float, default=0.996,
                            help="Teacher parameters are updated with EMA starting from this value")
        parser.add_argument("--param_momentum_final", type=float, default=0.996,
                            help="Teacher parameters are updated with EMA ending at this value")
        parser.add_argument("--param_momentum_epochs_end", type=int, default=10000,
                            help="The teacher parameters momentum reach its final value after this many epochs")
        parser.add_argument("--center_momentum", type=float, default=0.9,
                            help="momentum for updating the teacher softmax")

        # temperatures
        parser.add_argument("--set_temperature_using_ipr_init", type=smart_bool, default=True,
                            help="If true the student and teacher's temperatures are fixed so that IPR \
                            is equal to the ipr_init")
        parser.add_argument("--temperature_teacher_init", type=float, default=0.04,
                            help="Initial value of the temperature in the softmax of the teacher. \
                            Ignored if set_temperature_using_ipr_init = True")
        parser.add_argument("--temperature_student_init", type=float, default=0.1,
                            help="Initial value of the temperature in the softmax of the student.")
        parser.add_argument("--ipr_teacher_init", type=float, default=40.0,
                            help="The desired value of the initial IPR for the teacher. \
                            Ignored if set_temperature_using_ipr_init = True")
        parser.add_argument("--ipr_student_init", type=float, default=80.0,
                            help="The desired value of the initial IPR for the student. \
                            Ignored if set_temperature_using_ipr_init = True")

        # scheduler
        parser.add_argument("--max_epochs", type=int, default=1000, help="maximum number of training epochs")
        parser.add_argument("--warm_up_epochs", default=100, type=int,
                            help="Number of epochs for the linear learning-rate warm up.")
        parser.add_argument("--warm_down_epochs", default=500, type=int,
                            help="Number of epochs for the cosine decay.")

        parser.add_argument('--min_learning_rate', type=float, default=1e-5,
                            help="Target LR at the end of cosine protocol (smallest LR used during training).")
        parser.add_argument("--max_learning_rate", type=float, default=5e-4,
                            help="learning rate at the end of linear ramp (largest LR used during training).")

        parser.add_argument('--min_weight_decay', type=float, default=0.04,
                            help="Minimum value of the weight decay. It is used during the linear ramp.")
        parser.add_argument('--max_weight_decay', type=float, default=0.4,
                            help="Maximum Value of the weight decay. It is reached at the end of cosine protocol.")
        return parser

    @classmethod
    def get_default_params(cls) -> dict:
        parser = ArgumentParser()
        parser = DinoModel.add_specific_args(parser)
        args = parser.parse_args(args=[])
        return args.__dict__

    def forward(self, x) -> torch.Tensor:
        z, y = self.teacher(x)
        return y

    def shared_step(self, x):
        # step common to train_step and validation_step
        z, y = self.teacher(x)
        return z, y

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        with torch.no_grad():
            # Update the optimizer parameters
            lr = self.learning_rate_fn(self.current_epoch)
            wd = self.weight_decay_fn(self.current_epoch)
            for i, param_group in enumerate(self.optimizers().param_groups):
                param_group["lr"] = lr
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd
                else:
                    param_group["weight_decay"] = 0.0

            # this is data augmentation
            list_imgs, list_labels, list_metadata = batch

            # create the global and local crops
            list_of_minibatches = []
            for n in range(self.n_global_crops):
                list_of_minibatches.append(self.trsfm_train_global(list_imgs))
            for n in range(self.n_local_crops):
                list_of_minibatches.append(self.trsfm_train_local(list_imgs))

            # forward for teacher is inside the no-grad context
            z_t, y_t = self.shared_step(list_of_minibatches[:self.n_global_crops])
        # forward for student is outside the no-grad context
        z_s, y_s = self.student(list_of_minibatches)

        with torch.no_grad():
            if self.global_step == 0 and self.set_temperature_using_ipr_init:
                self.__update_temperatures__(
                    output_t=z_t, output_s=z_s, ipr_t=self.ipr_teacher_init, ipr_s=self.ipr_student_init)

        # loss
        loss: torch.Tensor
        prob_t: torch.Tensor
        prob_s: torch.Tensor
        loss, prob_t, prob_s = dino_loss(
            output_t=z_t,
            output_s=z_s,
            ncrops_t=self.n_global_crops,
            ncrops_s=self.n_global_crops+self.n_local_crops,
            temp_s=self.student_temperature,
            temp_t=self.teacher_temperature,
            center_t=self.center_teacher,
        )

        # gather stuff from all GPUs and update the teacher on all GPUs identically
        with torch.no_grad():
            # compute sample entropy
            tmp_entropy_sample_t = self.__normalized_stable_entropy__(prob_t).mean()  # shape: []
            tmp_entropy_sample_s = self.__normalized_stable_entropy__(prob_s).mean()  # shape: []

            # compute sample ipr
            tmp_ipr_sample_t = self.__inverse_participation_ratio__(prob_t).mean()  # shape: []
            tmp_ipr_sample_s = self.__inverse_participation_ratio__(prob_s).mean()  # shape: []

            # compute the pdf of the population
            tmp_population_t = prob_t.sum(dim=0)  # sum over samples --> shape: latent
            tmp_population_s = prob_s.sum(dim=0)  # sum over samples --> shape: latent

            # compute the teacher center
            tmp_empirical_center_teacher = z_t.mean(dim=0)  # sum over samples --> shape: latent

            # compute the local batch_size
            tmp_local_batch_size = torch.tensor(len(list_imgs), device=self.device, dtype=torch.float)

            # do the all_gather operations together
            tmp = [tmp_entropy_sample_t, tmp_entropy_sample_s,
                   tmp_ipr_sample_t, tmp_ipr_sample_s,
                   tmp_population_t, tmp_population_s,
                   tmp_empirical_center_teacher,
                   tmp_local_batch_size]
            world_tmp = self.all_gather(tmp)
            w_ent_t, w_ent_s, w_ipr_t, w_ipr_s, w_pop_t, w_pop_s, w_center_t, w_batch_size = world_tmp

            # update the population over a mini-batch
            if len(w_pop_s.shape) == 1 + len(tmp_population_s.shape):
                self.population_s.add_(w_pop_s.sum(dim=0))         # shape: latent
                self.population_t.add_(w_pop_t.sum(dim=0))         # shape: latent
                empirical_center_teacher = w_center_t.mean(dim=0)  # shape: latent
            else:
                self.population_s.add_(w_pop_s)        # shape: latent
                self.population_t.add_(w_pop_t)        # shape: latent
                empirical_center_teacher = w_center_t  # shape: latent

            # update teacher parameter and center
            center_momentum = self.teacher_center_momentum
            param_momentum = self.teacher_parameter_momentum_fn(self.current_epoch)
            self.__update_teacher_param__(p_momentum=param_momentum)
            self.__update_teacher_center__(empirical_center_teacher=empirical_center_teacher,
                                           c_momentum=center_momentum)

            # Finally I log interesting stuff
            self.log('train_loss', loss, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('train_loss', loss, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('weight_decay', wd, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('learning_rate', lr, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('teacher_param_momentum',
                     param_momentum, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('teacher_center_momentum',
                     center_momentum, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('teacher_temperature',
                     self.teacher_temperature, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('student_temperature',
                     self.student_temperature, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('ipr_studnet_init', self.ipr_student_init,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('ipr_teacher_init', self.ipr_teacher_init,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)

            # These are correct both if all_gather add an extra dimension or not (depends if trainer(strategy='dpp')).
            self.log('batch_size_per_gpu_train', w_batch_size.mean(), on_step=False, on_epoch=True, rank_zero_only=True)
            self.log('batch_size_total_train', w_batch_size.sum(), on_step=False, on_epoch=True, rank_zero_only=True)
            self.log('entropy_sample_t', w_ent_t.mean(), on_step=False, on_epoch=True, rank_zero_only=True)
            self.log('entropy_sample_s', w_ent_s.mean(), on_step=False, on_epoch=True, rank_zero_only=True)
            self.log('ipr_sample_t', w_ipr_t.mean(), on_step=False, on_epoch=True, rank_zero_only=True)
            self.log('ipr_sample_s', w_ipr_s.mean(), on_step=False, on_epoch=True, rank_zero_only=True)

        return loss

    def on_train_epoch_start(self) -> None:
        self.population_t.fill_ = 0.0
        self.population_s.fill_ = 0.0

    def on_train_epoch_end(self, unused=None) -> None:
        assert self.population_t.shape == self.population_s.shape == torch.Size([self.dim_out]), \
            "This should be 1D vector of size: {0}. Received {0}".format(self.dim_out, self.population_t.shape)

        tmp_t = self.population_t / torch.sum(self.population_t)  # shape: latent
        tmp_s = self.population_s / torch.sum(self.population_s)  # shape: latent

        # compute population entropy and population_ipr
        entropy_population_t = self.__normalized_stable_entropy__(tmp_t)
        entropy_population_s = self.__normalized_stable_entropy__(tmp_s)
        ipr_population_t = self.__inverse_participation_ratio__(tmp_t)
        ipr_population_s = self.__inverse_participation_ratio__(tmp_s)

        self.log('entropy_population_t', entropy_population_t,
                 on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
        self.log('entropy_population_s', entropy_population_s,
                 on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
        self.log('ipr_population_t', ipr_population_t,
                 on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
        self.log('ipr_population_s', ipr_population_s,
                 on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)

    def __update_temperatures__(self, output_t, output_s, ipr_t, ipr_s):

        ideal_temp_teacher_tmp = self.__find_ideal_temperature__(
            output=output_t,
            tau_init=torch.ones_like(self.teacher_temperature),
            desired_ipr=ipr_t)

        ideal_temp_student_tmp = self.__find_ideal_temperature__(
            output=output_s,
            tau_init=torch.ones_like(self.student_temperature),
            desired_ipr=ipr_s)

        w_ideal_temp_teacher, w_ideal_temp_student = self.all_gather(
            [ideal_temp_teacher_tmp, ideal_temp_student_tmp])

        if len(w_ideal_temp_student.shape) == 1 + len(ideal_temp_student_tmp.shape):
            ideal_temp_teacher = w_ideal_temp_teacher.mean(dim=0).clamp(min=1.0E-5, max=1.0E5)
            ideal_temp_student = w_ideal_temp_student.mean(dim=0).clamp(min=1.0E-5, max=1.0E5)
        else:
            ideal_temp_teacher = w_ideal_temp_teacher.clamp(min=1.0E-5, max=1.0E5)
            ideal_temp_student = w_ideal_temp_student.clamp(min=1.0E-5, max=1.0E5)

        assert ideal_temp_teacher.shape == ideal_temp_student.shape == torch.Size([1]), \
            "Expected torch.Size([1]). Received {0}, {1}".format(ideal_temp_teacher.shape, ideal_temp_student.shape)

        assert ideal_temp_teacher.isfinite() and ideal_temp_student.isfinite(), \
            "Ideal temperature is not finite. Received {0}, {1}".format(ideal_temp_teacher, ideal_temp_student)

        assert ideal_temp_teacher < ideal_temp_student, \
            "Error. teacher temperature {0} must be SMALLER than student temperature {1}.".format(ideal_temp_teacher,
                                                                                                  ideal_temp_student)
        self.teacher_temperature = ideal_temp_teacher
        self.student_temperature = ideal_temp_student

    @staticmethod
    def __find_ideal_temperature__(
            output: torch.Tensor,
            tau_init: torch.Tensor,
            desired_ipr: float,
            f_eps: float = 1E-3,
            eps: float = 1E-4) -> torch.Tensor:
        """
        Find the temperature that makes the IPR = desired_ipr.
        Solve this problem using bisection search.

        Args:
            output: the raw output of the student network of shape (batch, latent_dim)
            tau_init: initial guess for the temperature to use. A good guess make search conclude earlier.
            desired_ipr: float larger than 1.01. The desired IPR for a sample.
                IPR=1.0 means one-hot probabilities. Temperature tends to 0.0 if IPR tends to 1.0.
            f_eps: algorithm terminates if (IPR - desired_ipr).abs() < f_eps
            eps: algorithm terminates if the (tau_max - tau_min) < eps

        Return:
            tau, a torch.Tensor with the temperature value which makes the IPR close to the desired value.
        """

        def f(_temperature: torch.Tensor) -> torch.Tensor:
            """ This is the function I want to find the zero of. """
            _logit = output / _temperature
            _p = F.softmax(_logit, dim=-1)
            _ipr_sample = 1.0 / (_p * _p).sum(dim=-1)
            tmp = _ipr_sample.mean(dim=0) - desired_ipr
            assert torch.isfinite(tmp), \
                "Error. The function I want to find the zero of is not finite {0}".format(tmp)
            return tmp

        assert isinstance(desired_ipr, float) and desired_ipr > 1.01, \
            "Desired IPR must be a float > 1.01. Received {0}".format(desired_ipr)

        with torch.no_grad():
            n_max = 20
            n = 0
            a = b = tau_init
            fa = fb = f(a)

            if fb > 0:
                # ipr too high -> decrease temperature
                while (fb > 0) and (n < n_max):
                    n = n + 1
                    fa = fb
                    a = b
                    b = b * 0.5
                    fb = f(b)
            else:
                # ipr too low -> increase temperature
                while (fb < 0) and (n < n_max):
                    n = n + 1
                    fa = fb
                    a = b
                    b = b * 2.0
                    fb = f(b)

            if fa * fb < 0.0 and torch.isfinite(a) and torch.isfinite(b):
                # I have bracketed a zero. Now I can do bisection algorithm
                n_max = 20
                n = 0
                fc = fb
                while abs(a - b) > eps and fc.abs() > f_eps and n < n_max:
                    c = 0.5 * (a + b)
                    fc = f(c)
                    n = n + 1
                    if fa * fc < 0:
                        b = c
                        fb = fc
                    elif fb * fc < 0:
                        a = c
                        fa = fc
                    else:
                        raise Exception("something wrong happened: a,b,c=({0},{1},{2}) \
                        and fa,fb,fc=({3},{4},{5})".format(a, b, c, fa, fb, fc))
                return 0.5 * (a + b)
            else:
                print("WARNING. dynamic temperature routine did not find a finite solution.")
                return torch.tensor([float('inf')], device=output.device, dtype=torch.float)

    @staticmethod
    def __inverse_participation_ratio__(x):
        ipr = 1.0 / (x * x).sum(dim=-1)
        return ipr

    @staticmethod
    def __normalized_stable_entropy__(x):
        x_logx = x * x.log()
        tmp = torch.where(torch.isfinite(x_logx), x_logx, torch.zeros_like(x_logx))
        return -torch.sum(tmp, dim=-1) / numpy.log(float(tmp.shape[-1]))

    def __update_teacher_param__(self, p_momentum: float):
        if p_momentum < 1.0:
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data.mul_(p_momentum).add_((1 - p_momentum) * param_s.data.detach())

    def __update_teacher_center__(self, empirical_center_teacher: torch.Tensor, c_momentum: float):
        assert empirical_center_teacher.shape == torch.Size([self.dim_out]), \
            "Received {0}. Expected torch.Size([{1}])".format(empirical_center_teacher.shape, self.dim_out)
        self.center_teacher = c_momentum * self.center_teacher + (1.0-c_momentum) * empirical_center_teacher

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        regularized = []
        not_regularized = []
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        arg_for_optimizer = [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]

        # The real lr will be set in the training step
        # The weight_decay for the regularized group will be set in the training step
        if self.optimizer_type == 'adam':
            return torch.optim.Adam(arg_for_optimizer, betas=(0.9, 0.999), lr=0.0)
        elif self.optimizer_type == 'sgd':
            return torch.optim.SGD(arg_for_optimizer, momentum=0.9, lr=0.0)
        elif self.optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(arg_for_optimizer, alpha=0.99, lr=0.0)
        elif self.optimizer_type == 'lars':
            # for convnet with large batch_size
            return LARS(arg_for_optimizer, momentum=0.9, lr=0.0)
        else:
            # do adamw
            raise Exception("optimizer is misspecified")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        self.neptune_run_id = checkpoint.get("neptune_run_id", None)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        checkpoint["neptune_run_id"] = getattr(self.logger, "_run_short_id", None)
