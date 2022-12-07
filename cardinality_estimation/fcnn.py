import torch
import numpy as np
from query_representation.utils import *
from .dataset import QueryDataset, pad_sets, to_variable
from .nets import *
from .algs import *

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_
import wandb

class FCNN(NN):

    def init_dataset(self, samples, load_query_together,
            max_num_tables = -1,
            subplan_mask = None,
            load_padded_mscn_feats=False):
        ds = QueryDataset(samples, self.featurizer,
                load_query_together,
                max_num_tables = max_num_tables,
                load_padded_mscn_feats=False)
        return ds

    def _init_net(self, sample):
        num_features = len(sample[0])
        if self.subplan_level_outputs:
            n_output = 10
        else:
            n_output = 1
        net = SimpleRegression(num_features, n_output,
                self.num_hidden_layers, self.hidden_layer_size,
                )
        return net

    def print_init_training_info(self):
        pass
