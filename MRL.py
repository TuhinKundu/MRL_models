# code adapted from https://github.com/RAIVNLab/MRL/blob/7ccb42df6be05f3d21d0648aa03099bba46386bf/MRL.py

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from open_clip.loss import ClipLoss
'''
Loss function for Matryoshka Representation Learning 
'''


class MatryoshkaClipLoss(ClipLoss):
    def __init__(self, nesting_dim, loss_weights, **kwargs):
        super().__init__(**kwargs)
        self.nesting_dim = nesting_dim

        self.loss_weights = loss_weights

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        total_loss = 0
        for i, dim in enumerate(self.nesting_dim):
            img_dim_features =  torch.nn.functional.normalize(image_features[:,:dim], dim=-1)
            text_dim_features = torch.nn.functional.normalize(text_features[:,:dim], dim=-1)
            loss = super().forward(img_dim_features, text_dim_features, logit_scale)
            total_loss += loss * self.loss_weights[i]

        return total_loss


class Matryoshka_MLM_Loss(nn.Module):
    def __init__(self, vocab_size: int, relative_importance: List[float] = None, **kwargs):
        super(Matryoshka_MLM_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
        self.vocab_size = vocab_size
        # relative importance shape: [G]
        self.relative_importance = relative_importance

    def forward(self, output, target):
        # output shape: [G granularities, N batch size, C number of classes]
        # target shape: [N batch size]
        # Calculate losses for each output and stack them. This is still O(N)
        losses = torch.stack([self.criterion(output_i.view(-1, self.vocab_size), target) for output_i in output])

        # Set relative_importance to 1 if not specified
        rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(
            self.relative_importance)

        # Apply relative importance weights
        weighted_losses = rel_importance * losses

        return weighted_losses.sum()


class MRL_Linear_Layer(nn.Module):
    def __init__(self, nesting_list: List, vocab_size: int, efficient=False, **kwargs):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.out_dim = vocab_size
        self.efficient = efficient
        if self.efficient:
            setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.out_dim, **kwargs))
            self.nesting_list = [nesting_list[-1]]
        else:
            for i, num_feat in enumerate(self.nesting_list):
                setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.out_dim, **kwargs))

    def reset_parameters(self):
        if self.efficient:
            self.nesting_classifier_0.reset_parameters()
        else:
            for i in range(len(self.nesting_list)):
                getattr(self, f"nesting_classifier_{i}").reset_parameters()

    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:
                    nesting_logits += (
                    torch.matmul(x[:, :,:num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()),)
                else:
                    nesting_logits += (torch.matmul(x[:, :,:num_feat], (
                    self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias,)
            else:

                nesting_logits += (getattr(self, f"nesting_classifier_{i}")(x[:, :,:num_feat]),)

        return nesting_logits