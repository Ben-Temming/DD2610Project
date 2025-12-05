# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from .multiclass_rankup_net import MultiClass_RankUp_Net
from .bin_encoder import BinEncoder

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
import torch


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
        if self.num_classes is None:
            raise ValueError("num_classes must be specified for MultiClass_RankUp")
        
        self.ordinal_ranking = args.ordinal_ranking if hasattr(args, "ordinal_ranking") else True  # whether to use ordinal regression loss
        self.bin_strategy = args.bin_strategy if hasattr(args, "bin_strategy") else "equal_width"  # binning strategy
        self.ce_loss = CELoss()
        self.cls_consistency_loss = ClsConsistencyLoss()
        super().__init__(args, net_builder, tb_log, logger)
        
        # Initialize and fit BinEncoder using labeled training data
        self._setup_bin_encoder()

    def init(self, arc_ulb_loss_ratio, arc_loss_ratio, T, p_cutoff, hard_label):
        self.arc_ulb_loss_ratio = arc_ulb_loss_ratio
        self.arc_loss_ratio = arc_loss_ratio
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def _setup_bin_encoder(self):
        """
        Initialize and fit the BinEncoder using labeled training data.
        This converts continuous targets (e.g., ages) to discrete bin classes.
        Note: targets are already scaled to [0, 1] by the base class using the
        labeled training data.
        """
        self.bin_encoder = BinEncoder(
            num_classes=self.num_classes,
            strategy=self.bin_strategy
        )
        
        # Get labeled targets from the dataset (already scaled to [0, 1])
        lb_targets = self.dataset_dict["train_lb"].targets
        
        # Fit the bin encoder on scaled values
        self.bin_encoder.fit(lb_targets)
        
        # Log bin information (show both scaled and original values)
        self.print_fn("=" * 50)
        self.print_fn("BinEncoder initialized for MultiClass RankUp")
        self.print_fn(self.bin_encoder.get_bin_info(scaler=self.scaler, original_range=self.input_range))
        self.print_fn("=" * 50)


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

        # Convert continuous labels (ages) to bin class indices
        y_lb_bins = self.bin_encoder.transform(y_lb)

        # inference and calculate sup losses
        with self.amp_cm():
            # labeled prediction
            outs_x_lb = self.model(x_lb, use_arc=True, targets=y_lb)
            logits_x_lb = outs_x_lb["logits"]
            feats_x_lb = outs_x_lb["feat"]
            logits_arc_x_lb = outs_x_lb["logits_arc"]
            # Use binned labels instead of raw targets
            arc_y_lb = y_lb_bins

            # unlabeled weak prediction
            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb_w = self.model(x_ulb_w, use_arc=True)
            feats_x_ulb_w = outs_x_ulb_w["feat"]
            logits_arc_x_ulb_w = outs_x_ulb_w["logits_arc"]
            
            # For CORAL, use sigmoid to get cumulative probabilities
            # For standard classification, use softmax
            if self.ordinal_ranking:
                # CORAL: sigmoid gives P(Y > k) for each threshold k
                probs_x_ulb_w = torch.sigmoid(logits_arc_x_ulb_w.detach())
            else:
                probs_x_ulb_w = self.compute_prob(logits_arc_x_ulb_w.detach())
            self.bn_controller.unfreeze_bn(self.model)

            # unlabeled strong prediction
            outs_x_ulb_s = self.model(x_ulb_s, use_arc=True)
            logits_x_ulb_s = outs_x_ulb_s["logits_arc"]
            feats_x_ulb_s = outs_x_ulb_s["feat"]

            feat_dict = {"x_lb": feats_x_lb, "x_ulb_w": feats_x_ulb_w, "x_ulb_s": feats_x_ulb_s}

            # supervised loss (labeled data)
            sup_loss = self.reg_loss(logits_x_lb, y_lb, reduction="mean")

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
                # arc_y_lb is now bin class indices (from BinEncoder), not one-hot
                hard_cum_targets = levels_from_labelbatch(arc_y_lb, num_classes=self.num_classes)
                # Move to same device as logits
                hard_cum_targets = hard_cum_targets.to(logits_arc_x_lb.device)
                arc_sup_loss = coral_loss(logits_arc_x_lb, hard_cum_targets)

                # -- unsupervised -> use cumulative probabilities as soft targets --
                # For CORAL, probs_x_ulb_w is already sigmoid output (cumulative probs)
                # Compute mask based on confidence: how close each cumulative prob is to 0 or 1
                confidence = torch.abs(probs_x_ulb_w - 0.5) * 2  # Map [0,1] to confidence [0,1]
                avg_confidence = confidence.mean(dim=1)  # Average confidence across all thresholds
                mask = (avg_confidence >= self.p_cutoff).float()
                
                # Use cumulative probabilities directly as soft pseudo-labels
                arc_pseudo_label = probs_x_ulb_w
                
                arc_unsup_loss = soft_coral_loss(logits_x_ulb_s, arc_pseudo_label, mask=mask)

            else:
                # -- CE loss for multi-class (standard) classification --
                # Compute mask based on max class probability
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
                
                # Generate pseudo-labels using pseudo label hook
                arc_pseudo_label = self.call_hook(
                    "gen_ulb_targets",
                    "PseudoLabelingHook",
                    logits=probs_x_ulb_w,
                    use_hard_label=self.use_hard_label,
                    T=self.T,
                    softmax=False,
                )
                
                # arc_y_lb is bin class indices (Long tensor) from BinEncoder
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
            SSL_Argument("--num_classes", int, 8),  # number of bins for multiclass RankUp
            SSL_Argument("--ordinal_ranking", str2bool, True),  # whether to use ordinal regression loss (CORAL)
            SSL_Argument("--bin_strategy", str, "equal_width"),  # binning strategy: "equal_width" or "quantile"
        ]
