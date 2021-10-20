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

    def init_dataset(self, samples):
        ds = QueryDataset(samples, self.featurizer,
                False)
        return ds

    def _init_net(self, sample):
        num_features = len(sample[0])
        net = SimpleRegression(num_features, 1,
                self.num_hidden_layers, self.hidden_layer_size)
        return net

    def print_init_training_info(self):
        pass
