# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from .multiclass_rankup_net import MultiClass_RankUp_Net

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook

from semilearn.core.criterions import CELoss, ClsConsistencyLoss
"""
Install coral-pytorch package:
- pip install coral_pytorch
- pip install -r requirements.txt
"""
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch
# from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F


"""
I think this is equivalent to CORAL when mask=None
I think this could also be implemented using coral_loss without reduction and then applying mask:
    loss = (coral_loss(reduction='none').mean(dim=1)*mask).mean()
"""
def soft_coral_loss(logits, soft_targets, mask=None):
    """
    Compute CORAL loss for soft/pseudo targets.

    Args:
        logits: 
        soft_targets: soft/pseudo labels (probabilities), cumulative targets, with shape [batch_size, num_classes]
    """
    num_classes = logits.size(1)
    batch_size = logits.size(0)

    # Note: coral_loss() expects hard labels, so not directly usable for soft labels 
    # (it could be used if we threshold pseudo-labels to get hard labels, but that loses information)
    # BCE is equivalent of CORAL
    loss = F.binary_cross_entropy_with_logits(logits, # strong augmented logits
                                            soft_targets, # soft/pseudo targets from weak augmented
                                            reduction="none")  
    
    # apply mask, only consider confident pseudo-labels
    if mask is not None:
        loss = loss.mean(dim=1) * mask  # sample-wise mean, mask unconfident samples
    return loss.mean() # mean over batch


@ALGORITHMS.register("multiclass_rankup")
class MultiClass_RankUp(AlgorithmBase):
    """
    RankUp algorithm (https://arxiv.org/abs/2410.22124).

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - arc_ulb_loss_ratio (`float`):
            Weight for unsupervised loss in Arc
        - arc_loss_ratio (`float`):
            Weight for Arc loss
        - T (`float`):
            Temperature for pseudo-label sharpening
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - hard_label (`bool`, *optional*, default to `False`):
            If True, targets have [Batch size] shape with int values. If False, the target is vector
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.init(
            arc_ulb_loss_ratio=args.arc_ulb_loss_ratio,
            arc_loss_ratio=args.arc_loss_ratio,
            T=args.T,
            p_cutoff=args.p_cutoff,
            hard_label=args.hard_label,
        )
        self.num_classes = args.num_classes if hasattr(args, "num_classes") else None  # specify number of classes for multi-class classification
        self.ordinal_ranking = args.ordinal_ranking if hasattr(args, "ordinal_ranking") else True  # whether to use ordinal regression loss
        self.ce_loss = CELoss()
        self.cls_consistency_loss = ClsConsistencyLoss()
        super().__init__(args, net_builder, tb_log, logger)

    def init(self, arc_ulb_loss_ratio, arc_loss_ratio, T, p_cutoff, hard_label):
        self.arc_ulb_loss_ratio = arc_ulb_loss_ratio
        self.arc_loss_ratio = arc_loss_ratio
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_hooks(self):
        super().set_hooks()
        # reset PseudoLabelingHook hook
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")

    def set_model(self, **kwargs):
        """
        overwrite the initialize model function
        """
        model = super().set_model(**kwargs)
        model = MultiClass_RankUp_Net(model, num_classes=self.num_classes, ordinal_ranking=self.ordinal_ranking) # specify num_classes
        return model

    def set_ema_model(self, **kwargs):
        """
        overwrite the initialize ema model function
        """
        ema_model = self.net_builder(pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path, **kwargs)
        ema_model = MultiClass_RankUp_Net(ema_model, num_classes=self.num_classes, ordinal_ranking=self.ordinal_ranking) # specify num_classes
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        self.idx_ulb = idx_ulb

        # inference and calculate sup losses
        with self.amp_cm():
            # labeled prediction
            outs_x_lb = self.model(x_lb, use_arc=True, targets=y_lb)
            logits_x_lb = outs_x_lb["logits"]
            feats_x_lb = outs_x_lb["feat"]
            logits_arc_x_lb = outs_x_lb["logits_arc"]
            arc_y_lb = outs_x_lb["targets_arc"]

            # unlabeled weak prediction
            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb_w = self.model(x_ulb_w, use_arc=True)
            feats_x_ulb_w = outs_x_ulb_w["feat"]
            logits_arc_x_ulb_w = outs_x_ulb_w["logits_arc"]
            probs_x_ulb_w = self.compute_prob(logits_arc_x_ulb_w.detach())
            self.bn_controller.unfreeze_bn(self.model)

            # unlabeled strong prediction
            outs_x_ulb_s = self.model(x_ulb_s, use_arc=True)
            logits_x_ulb_s = outs_x_ulb_s["logits_arc"]
            feats_x_ulb_s = outs_x_ulb_s["feat"]

            feat_dict = {"x_lb": feats_x_lb, "x_ulb_w": feats_x_ulb_w, "x_ulb_s": feats_x_ulb_s}

            # supervised loss (labeled data)
            sup_loss = self.reg_loss(logits_x_lb, y_lb, reduction="mean")

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            arc_pseudo_label = self.call_hook(
                "gen_ulb_targets",
                "PseudoLabelingHook",
                logits=probs_x_ulb_w,
                use_hard_label=self.use_hard_label,
                T=self.T,
                softmax=False,
            )

            # since not computing pairwise differences for multi-class, the following is equivalent to FixMatch loss
            if self.ordinal_ranking:
                """
                -- Multi-class ordinal regression loss -- 
                penalize more for larger misclassifications, less for smaller ones

                CORAL (COnsistent RAnk Logits): https://arxiv.org/abs/1901.07884 
                using the CORAL pytorch implementation from: https://raschka-research-group.github.io/coral-pytorch/
                - levels = levels_from_labelbatch(class_labels, num_classes)
                - loss = coral_loss(logits, levels)
                """
                # -- supervised -> hard cumulative targets --
                hard_cum_targets = levels_from_labelbatch(arc_y_lb.argmax(dim=1), num_classes=self.num_classes)
                arc_sup_loss = coral_loss(logits_arc_x_lb, hard_cum_targets)

                # -- unsupervised -> soft/pseudo cumulative targets --
                # Note: coral_loss() expects hard labels, so not directly usable for soft labels 
                # (it could be used if we threshold pseudo-labels to get hard labels, but that loses information)
                # BCE is equivalent
                # self.bce_with_logits_loss = BCEWithLogitsLoss(reduction="none") # reduction none to apply mask, then mean
                # arc_unsup_loss = self.bce_with_logits_loss(logits_x_ulb_s, arc_pseudo_label) # strong augmented logits vs soft/pseudo targets from weak augmented
                # apply mask, only consider confident pseudo-labels
                # if mask is not None:
                #     arc_unsup_loss = arc_unsup_loss.mean(dim=1) * mask  # sample-wise mean, mask unconfident samples
                # arc_unsup_loss = arc_unsup_loss.mean()  # mean over batch

                arc_unsup_loss = soft_coral_loss(logits_x_ulb_s, arc_pseudo_label, mask=mask) # wrapper function for soft coral loss
                # Note: arc_pseudo_label should be cumulative soft targets, because generated from CoralLayer 

            else:
                # -- CE loss for multi-class (standard) classification --
                arc_sup_loss = self.ce_loss(logits_arc_x_lb, arc_y_lb, reduction="mean") # supervised Arc loss
                arc_unsup_loss = self.cls_consistency_loss(logits_x_ulb_s, arc_pseudo_label, "ce", mask=mask) # unsupervised Arc loss

            reg_loss = sup_loss + self.ulb_loss_ratio
            arc_loss = arc_sup_loss + self.arc_ulb_loss_ratio * arc_unsup_loss
            total_loss = reg_loss + self.arc_loss_ratio * arc_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=(sup_loss + arc_sup_loss).item(), unsup_loss=arc_unsup_loss.item(), total_loss=total_loss.item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--arc_ulb_loss_ratio", float, 1.0),
            SSL_Argument("--arc_loss_ratio", float, 1.0),
            SSL_Argument("--T", float, 0.5),
            SSL_Argument("--p_cutoff", float, 0.95),
            SSL_Argument("--hard_label", str2bool, True),
            SSL_Argument("--num_classes", int, 2),  # add num_classes for multiclass RankUp
            SSL_Argument("--ordinal_ranking", str2bool, True),  # whether to use ordinal regression loss
        ]
