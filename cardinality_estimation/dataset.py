import torch
from torch.utils import data
from torch.autograd import Variable
from collections import defaultdict
import numpy as np
import time
import copy

from query_representation.utils import *

import pdb

def to_variable(arr, use_cuda=True, requires_grad=False):
    if isinstance(arr, list) or isinstance(arr, tuple):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        arr = Variable(torch.from_numpy(arr), requires_grad=requires_grad)
    else:
        arr = Variable(arr, requires_grad=requires_grad)
    return arr

def _handle_set_padding(features, max_set_vals):

    if len(features) == 0:
        return None, None

    features = np.vstack(features)
    num_pad = max_set_vals - features.shape[0]
    assert num_pad >= 0

    mask = np.ones_like(features).mean(1, keepdims=True)
    features = np.pad(features, ((0, num_pad), (0, 0)), 'constant')
    mask = np.pad(mask, ((0, num_pad), (0, 0)), 'constant')
    features = np.expand_dims(features, 0)
    mask = np.expand_dims(mask, 0)

    return features, mask

def pad_sets(all_table_features, all_pred_features,
        all_join_features, maxtabs, maxpreds, maxjoins):

    tf = []
    pf = []
    jf = []
    tm = []
    pm = []
    jm = []

    assert len(all_table_features) == len(all_pred_features) == len(all_join_features)

    for i in range(len(all_table_features)):
        table_features = all_table_features[i]
        pred_features = all_pred_features[i]
        join_features = all_join_features[i]

        pred_features, predicate_mask = _handle_set_padding(pred_features,
                maxpreds)
        table_features, table_mask = _handle_set_padding(table_features,
                maxtabs)
        join_features, join_mask = _handle_set_padding(join_features,
                maxjoins)

        if table_features is not None:
            tf.append(table_features)
            tm.append(table_mask)

        if pred_features is not None:
            pf.append(pred_features)
            pm.append(predicate_mask)

        if join_features is not None:
            jf.append(join_features)
            jm.append(join_mask)

    tf = to_variable(tf,
            requires_grad=False).float().squeeze()
    extra_dim = len(tf.shape)-1
    tm = to_variable(tm,
            requires_grad=False).byte().squeeze().unsqueeze(extra_dim)

    pf = to_variable(pf,
            requires_grad=False).float().squeeze()
    extra_dim = len(pf.shape)-1
    pm = to_variable(pm,
            requires_grad=False).byte().squeeze().unsqueeze(extra_dim)

    jf = to_variable(jf,
            requires_grad=False).float().squeeze()
    extra_dim = len(jf.shape)-1

    jm = to_variable(jm,
            requires_grad=False).byte().squeeze().unsqueeze(extra_dim)

    return tf, pf, jf, tm, pm, jm

class QueryDataset(data.Dataset):
    def __init__(self, samples, featurizer,
            load_query_together, load_padded_mscn_feats=False):
        '''
        @samples: [] sqlrep query dictionaries, which represent a query and all
        of its subplans.
        @load_query_together: each sample will be a list of all the feature
        vectors belonging to all the subplans of a query.
        '''
        self.load_query_together = load_query_together
        self.load_padded_mscn_feats = load_padded_mscn_feats

        self.featurizer = featurizer

        # shorthands
        self.ckey = self.featurizer.ckey
        self.minv = self.featurizer.min_val
        self.maxv = self.featurizer.max_val
        self.feattype = self.featurizer.featurization_type

        # TODO: we may want to avoid this, and convert them on the fly. Just
        # keep some indexing information around.

        self.X, self.Y, self.info = self._get_feature_vectors(samples)
        self.num_samples = len(self.X)

    def _get_query_features(self, qrep, dataset_qidx,
            query_idx):
        '''
        @qrep: qrep dict.
        '''
        X = []
        Y = []
        sample_info = []

        # now, we will generate the actual feature vectors over all the
        # subplans. Order matters --- dataset idx will be specified based on
        # order.
        node_names = list(qrep["subset_graph"].nodes())
        if SOURCE_NODE in node_names:
            node_names.remove(SOURCE_NODE)
        node_names.sort()

        for node_idx, node in enumerate(node_names):
            x,y = self.featurizer.get_subplan_features(qrep,
                    node)

            if self.featurizer.featurization_type == "set" \
                and self.load_padded_mscn_feats:
                tf,pf,jf,tm,pm,jm = \
                    pad_sets([x["table"]], [x["pred"]], [x["join"]],
                            self.featurizer.max_tables, self.featurizer.max_preds,
                            self.featurizer.max_joins)
                x["table"] = tf
                x["join"] = jf
                x["pred"] = pf
                # relevant masks
                x["tmask"] = tm
                x["pmask"] = pm
                x["jmask"] = jm

                # x["flow"] remains the correct vector

            X.append(x)
            Y.append(y)

            cur_info = {}
            cur_info["num_tables"] = len(node)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["query_idx"] = query_idx
            sample_info.append(cur_info)

        return X,Y,sample_info

    def _get_feature_vectors(self, samples):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.
        '''
        start = time.time()
        X = []
        Y = []
        sample_info = []
        qidx = 0

        for i, qrep in enumerate(samples):
            x,y,cur_info = self._get_query_features(qrep, qidx, i)
            qidx += len(y)
            X += x
            Y += y
            sample_info += cur_info

        print("Extracting features took: ", time.time() - start)

        if self.featurizer.featurization_type == "combined":
            X = to_variable(X, requires_grad=False).float()
        elif self.featurizer.featurization_type == "set":
            # don't need to do anything, since padding+masks is handled later
            pass

        Y = to_variable(Y, requires_grad=False).float()

        return X,Y,sample_info

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        '''
        if self.load_query_together:
            assert False, "needs to be implemented"
            start_idx = self.start_idxs[index]
            end_idx = start_idx + self.idx_lens[index]
            if self.feattype == "combined":
                return self.X[start_idx:end_idx], self.Y[start_idx:end_idx], \
                        self.info[start_idx:end_idx]
        else:
            return self.X[index], self.Y[index], self.info[index]
