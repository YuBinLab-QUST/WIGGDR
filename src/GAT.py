from typing import Dict

import numpy as np
import torch
from torch import nn

from labml import lab, monit, tracker, experiment
from labml.configs import BaseConfigs, option, calculate
from labml.utils import download
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_nn.graphs.gat import GraphAttentionLayer
from labml_nn.optimizers.configs import OptimizerConfigs


class GAT(Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=False, dropout=dropout)
        self.activation = nn.ELU()
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x, adj_mat)
