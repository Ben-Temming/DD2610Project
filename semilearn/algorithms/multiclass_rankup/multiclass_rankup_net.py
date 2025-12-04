# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import torch
import torch.nn as nn

from semilearn.nets.utils import init_weights
from coral_pytorch.layers import CoralLayer


class MultiClass_RankUp_Net(nn.Module):
    """
    MultiClass_RankUp_Net implementation.

    Attributes:
        backbone (nn.Module): The underlying backbone model.
        num_features (int): Number of features from the model's hidden layer.
        arc_classifier (nn.Linear): Linear layer for Auxiliary Ranking Classifier (ARC) with two output classes.
        num_classes (int): Number of classes for multi-class classification (default is 2 = Rankup).
        ordinal_ranking (bool): Whether to use ordinal ranking with CORAL layer or just normal classification.
    """

    def __init__(self, backbone, num_classes=2, ordinal_ranking=True):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.num_classes = num_classes # number of classes for multi-class classification
        self.ordinal_ranking = ordinal_ranking
        # Note: if the number of classes is 2, it behaves similarly to the original RankUp but without pairwise comparisons, just one sample at a time

        # Auxiliary Ranking Classifier (ARC)
        if self.ordinal_ranking:
            """
            CORAS layer to enforce ordinal ranking
            difference from a linear layer: weights are shared among outputs to enforce ranking consistency
            reference: https://arxiv.org/abs/1901.07884, 3.2.2. Lossfunction, 3.2.3. Theoretical guarantees for classifier consistency
            """
            self.arc_classifier = CoralLayer(self.num_features, self.num_classes)
        else:
            # linear layer, no ordinal ranking
            self.arc_classifier = nn.Linear(self.num_features, self.num_classes)
        self.arc_classifier.apply(init_weights)

    def forward(self, x, use_arc=False, targets=None, **kwargs):
        if not use_arc:
            return self.backbone(x, **kwargs)
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        logits_arc = self.arc_classifier(feat)
        # logits_mat, targets_mat = self.compute_rank_logits(logits_arc, targets) 
        # return {"logits": logits, "logits_arc": logits_mat, "feat": feat, "targets_arc": targets_mat}

        # return predicted logits and targets 
        # compute_rank_logits() not used -> not computing pairwise differences (as in rankup) for multi-class classification
        return {"logits": logits, "logits_arc": logits_arc, "feat": feat, "targets_arc": targets}
        
    # Note: not used in multi-class classification, it only use one sample at a time, not pairs
    def compute_rank_logits(self, logits, targets=None):
        logits_mat = logits.unsqueeze(dim=0) - logits.unsqueeze(dim=1) # pairwise distances between samples' logits
        logits_mat = logits_mat.flatten(0, 1)
        if targets is not None:
            targets_mat = (1 + torch.sign(targets.unsqueeze(dim=0) - targets.unsqueeze(dim=1))) / 2
            targets_mat = targets_mat.flatten(0, 1)
            # one-hot encode the targets_mat
            targets_onehot = torch.zeros((targets_mat.shape[0], 2)).to(targets_mat.device)
            targets_onehot[:, 0] = targets_mat
            targets_onehot[:, 1] = 1 - targets_mat
            # for each pair of columns/outputs -> one hot encode can be [1,0], [0.5, 0.5], [0,1]
            return logits_mat, targets_onehot
        return logits_mat, None

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix="backbone.")
        return matcher
