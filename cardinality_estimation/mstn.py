import torch
import numpy as np
from query_representation.utils import *
from .dataset import QueryDataset, pad_sets, to_variable
from .nets import *
from .algs import *
# import cardinality_estimation.algs as algs

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_
import wandb

class MSTN(NN):

    def init_dataset(self, samples, load_query_together,
            max_num_tables = -1,
            load_padded_mscn_feats=False,
            subplan_mask=None):

        ds = QueryDataset(samples, self.featurizer,
                load_query_together,
                max_num_tables = max_num_tables,
                load_padded_mscn_feats=load_padded_mscn_feats)

        return ds

    def _init_net(self, sample):
        if self.load_query_together:
            sample = sample[0]

        if len(sample[0]["table"]) == 0:
            sfeats = 0
        else:
            sfeats = len(sample[0]["table"][0])

        if len(sample[0]["pred"]) == 0:
            pfeats = 0
        else:
            pfeats = len(sample[0]["pred"][0])

        if len(sample[0]["join"]) == 0:
            jfeats = 0
        else:
            jfeats = len(sample[0]["join"][0])

        if self.subplan_level_outputs:
            n_out = 10
        else:
            n_out = 1

        if self.featurizer.ynormalization in ["selectivity-log"]:
            use_sigmoid = False
        elif "whitening" in self.featurizer.ynormalization:
            use_sigmoid = False
        else:
            use_sigmoid = True

        if self.loss_func_name == "flowloss":
            assert False, "need to fix dataloader"
        else:
            net = CardSetTransformer(sfeats,
                    pfeats, jfeats,
                    len(sample[0]["flow"]),
                    self.hidden_layer_size,
                    n_out=n_out,
                    num_hidden_layers = self.num_hidden_layers,
                    dropouts=[self.inp_dropout, self.hl_dropout,
                        self.comb_dropout],
                    use_sigmoid = use_sigmoid)

        return net
