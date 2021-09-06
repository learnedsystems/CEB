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

        pred_features = np.vstack(pred_features)
        num_pad = maxpreds - pred_features.shape[0]
        assert num_pad >= 0
        # if num_pad < 0:
            # print(num_pad)
            # pdb.set_trace()

        predicate_mask = np.ones_like(pred_features).mean(1, keepdims=True)
        pred_features = np.pad(pred_features, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        pred_features = np.expand_dims(pred_features, 0)
        predicate_mask = np.expand_dims(predicate_mask, 0)

        # do same for table, and joins
        table_features = np.vstack(table_features)
        num_pad = maxtabs - table_features.shape[0]

        assert num_pad >= 0

        table_mask = np.ones_like(table_features).mean(1, keepdims=True)
        table_features = np.pad(table_features, ((0, num_pad), (0, 0)), 'constant')
        table_mask = np.pad(table_mask, ((0, num_pad), (0, 0)), 'constant')
        table_features = np.expand_dims(table_features, 0)
        table_mask = np.expand_dims(table_mask, 0)

        join_features = np.vstack(join_features)
        num_pad = maxjoins - join_features.shape[0]

        assert num_pad >= 0

        join_mask = np.ones_like(join_features).mean(1, keepdims=True)
        join_features = np.pad(join_features, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_features = np.expand_dims(join_features, 0)
        join_mask = np.expand_dims(join_mask, 0)

        tf.append(table_features)
        pf.append(pred_features)
        jf.append(join_features)
        tm.append(table_mask)
        pm.append(predicate_mask)
        jm.append(join_mask)

    tf = to_variable(tf,
            requires_grad=False).float().squeeze()
    pf = to_variable(pf,
            requires_grad=False).float().squeeze()
    jf = to_variable(jf,
            requires_grad=False).float().squeeze()
    extra_dim = len(jf.shape)-1

    tm = to_variable(tm,
            requires_grad=False).float().squeeze().unsqueeze(extra_dim)
    pm = to_variable(pm,
            requires_grad=False).float().squeeze().unsqueeze(extra_dim)
    jm = to_variable(jm,
            requires_grad=False).float().squeeze().unsqueeze(extra_dim)

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
        if self.feattype == "combined":
            X = []
        else:
            X = defaultdict(list)

        Y = []
        sample_info = []

        node_data = qrep["join_graph"].nodes(data=True)

        table_feat_dict = {}
        pred_feat_dict = {}
        edge_feat_dict = {}

        # iteration order doesn't matter
        for node, info in node_data:
            if SOURCE_NODE in node_data:
                continue

            cards = qrep["subset_graph"].nodes()[(node,)]
            if "sample_bitmap" in cards:
                bitmap = cards["sample_bitmap"]
            else:
                bitmap = None

            table_features = self.featurizer.get_table_features(info["real_name"],
                    bitmap_dict=bitmap)
            table_feat_dict[node] = table_features

            # TODO: pass in the cardinality as well.
            heuristic_est = None
            if self.featurizer.heuristic_features:
                node_key = tuple([node])
                cards = qrep["subset_graph"].nodes()[node_key][self.ckey]
                if "total" in cards:
                    total = cards["total"]
                else:
                    total = None
                heuristic_est = self.featurizer.normalize_val(cards["expected"],
                        total)

            if len(info["pred_cols"]) == 0:
                pred_features = np.zeros(self.featurizer.pred_features_len)
            else:
                pred_features = self.featurizer.get_pred_features(info["pred_cols"][0],
                        info["pred_vals"][0], info["pred_types"][0],
                        pred_est=heuristic_est)

            assert len(pred_features) == self.featurizer.pred_features_len
            pred_feat_dict[node] = pred_features

        edge_data = qrep["join_graph"].edges(data=True)
        for edge in edge_data:
            info = edge[2]
            edge_features = self.featurizer.get_join_features(info["join_condition"])
            edge_key = (edge[0], edge[1])
            edge_feat_dict[edge_key] = edge_features

        # now, we will generate the actual feature vectors over all the
        # subqueries. Order matters - dataset idx will be specified based on
        # order.
        node_names = list(qrep["subset_graph"].nodes())
        if SOURCE_NODE in node_names:
            node_names.remove(SOURCE_NODE)
        node_names.sort()

        for node_idx, nodes in enumerate(node_names):
            info = qrep["subset_graph"].nodes()[nodes]
            pg_est = info[self.ckey]["expected"]
            true_val = info[self.ckey]["actual"]
            if "total" in info[self.ckey]:
                total = info[self.ckey]["total"]
            else:
                total = None

            pred_features = np.zeros(self.featurizer.pred_features_len)
            table_features = np.zeros(self.featurizer.table_features_len)
            join_features = np.zeros(len(self.featurizer.joins))

            # these are base tables within a join (or node) in the subset
            # graph
            for node in nodes:
                # no overlap between these arrays
                pred_features += pred_feat_dict[node]
                table_features += table_feat_dict[node]

            if self.featurizer.heuristic_features:
                assert pred_features[-1] == 0.00
                pred_features[-1] = self.featurizer.normalize_val(pg_est, total)

            # TODO: optimize...
            for node1 in nodes:
                for node2 in nodes:
                    if (node1, node2) in edge_feat_dict:
                        join_features += edge_feat_dict[(node1, node2)]

            if self.featurizer.flow_features:
                if "pred_types" in info:
                    cmp_op = info["pred_types"][0]
                else:
                    cmp_op = None
                flow_features = self.featurizer.get_flow_features(nodes,
                        qrep["subset_graph"], qrep["template_name"],
                        qrep["join_graph"], cmp_op)
                # heuristic estimate for the cardinality of this node
                flow_features[-1] = pred_features[-1]
            else:
                flow_features = []

            # now, store features
            if self.feattype == "combined":
                comb_feats = []
                if self.featurizer.table_features:
                    comb_feats.append(table_features)
                if self.featurizer.join_features:
                    comb_feats.append(join_features)
                if self.featurizer.pred_features:
                    comb_feats.append(pred_features)
                if self.featurizer.flow_features:
                    comb_feats.append(flow_features)
                assert len(comb_feats) > 0
                X.append(np.concatenate(comb_feats))
            elif self.feattype == "set":
                X["table"].append(table_features)
                X["join"].append(join_features)
                X["pred"].append(pred_features)
                X["flow"].append(flow_features)

            Y.append(self.featurizer.normalize_val(true_val, total))
            cur_info = {}
            cur_info["num_tables"] = len(nodes)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["node_idx"] = node_idx
            cur_info["query_idx"] = query_idx

            sample_info.append(cur_info)

        return X,Y,sample_info

    def _get_query_features_set(self, qrep, dataset_qidx,
            query_idx):
        # FIXME: initial part of this function is same as
        # _get_query_features_combined; decompose

        X = []
        Y = []
        sample_info = []

        node_data = qrep["join_graph"].nodes(data=True)

        table_feat_dict = {}
        pred_feat_dict = {}
        edge_feat_dict = {}

        # iteration order doesn't matter
        for node, info in node_data:
            if SOURCE_NODE in node_data:
                continue
            cards = qrep["subset_graph"].nodes()[(node,)]
            if "sample_bitmap" in cards:
                bitmap = cards["sample_bitmap"]
            else:
                bitmap = None
            table_features = self.featurizer.get_table_features(info["real_name"],
                    bitmap_dict=bitmap)
            table_feat_dict[node] = table_features

            heuristic_est = None
            if self.featurizer.heuristic_features:
                node_key = tuple([node])
                cards = qrep["subset_graph"].nodes()[node_key]["cardinality"]
                if "total" in cards:
                    total = cards["total"]
                else:
                    total = None
                heuristic_est = self.featurizer.normalize_val(cards["expected"],
                        total)

            if len(info["pred_cols"]) != 0:
                pred_features = \
                    self.featurizer.get_pred_features(info["pred_cols"][0],
                            info["pred_vals"][0], info["pred_types"][0],
                            pred_est=heuristic_est)
                pred_feat_dict[node] = pred_features


        edge_data = qrep["join_graph"].edges(data=True)
        for edge in edge_data:
            info = edge[2]
            edge_features = \
                self.featurizer.get_join_features(info["join_condition"])
            edge_key = (edge[0], edge[1])
            edge_feat_dict[edge_key] = edge_features

        # now, we will generate the actual feature vectors over all the
        # subplans. Order matters --- dataset idx will be specified based on
        # order.
        node_names = list(qrep["subset_graph"].nodes())
        if SOURCE_NODE in node_names:
            node_names.remove(SOURCE_NODE)
        node_names.sort()

        for node_idx, nodes in enumerate(node_names):
            info = qrep["subset_graph"].nodes()[nodes]
            pg_est = info["cardinality"]["expected"]
            total = 0.0
            sample_heuristic_est = self.featurizer.normalize_val(pg_est, total)
            true_val = info["cardinality"]["actual"]
            if "total" in info["cardinality"]:
                total = info["cardinality"]["total"]
            else:
                total = None

            table_features = []
            pred_features = []
            join_features = []

            # these are base tables within a join (or node) in the subset
            # graph
            for node in nodes:
                table_features.append(table_feat_dict[node])
                if node not in pred_feat_dict:
                    continue
                # pred_features.append(pred_feat_dict[node])
                pfeats = copy.deepcopy(pred_feat_dict[node])

                # PN: adding subplan's heuristic estimate to every predicate as
                # extra feature; this is actually a global feature that could
                # be added to flow_features instead / but if we don't use them,
                # here, we just add the same feature to every predicate vector
                assert pfeats[-1] == 0.0
                pfeats[-1] = sample_heuristic_est
                pred_features.append(pfeats)

            for node1 in nodes:
                for node2 in nodes:
                    if (node1, node2) in edge_feat_dict:
                        join_features.append(edge_feat_dict[(node1, node2)])

            if len(join_features) == 0:
                empty_feats = np.zeros(len(self.featurizer.join_featurizer))
                join_features.append(empty_feats)

            if len(pred_features) == 0:
                empty_feats = np.zeros(self.featurizer.max_pred_len)
                # empty_feats = np.zeros(self.featurizer.pred_features_len)
                pred_features.append(empty_feats)

            if self.featurizer.flow_features:
                if "pred_types" in info:
                    cmp_op = info["pred_types"][0]
                else:
                    cmp_op = None
                flow_features = self.featurizer.get_flow_features(nodes,
                        qrep["subset_graph"], qrep["template_name"],
                        qrep["join_graph"], cmp_op)

                # heuristic estimate for the cardinality of this node
                if self.featurizer.heuristic_features:
                    assert flow_features[-1] == 0.0
                    flow_features[-1] = sample_heuristic_est
            else:
                flow_features = []

            # now, store features
            x = {}

            if self.load_padded_mscn_feats:
                # print(len(table_features))
                # assert len(table_features) == 1
                table_features,pred_features,join_features,tm,pm,jm = \
                    pad_sets([table_features], [pred_features], [join_features],
                            self.featurizer.max_tables, self.featurizer.max_preds,
                            self.featurizer.max_joins)
                x["table"] = table_features
                x["join"] = join_features
                x["pred"] = pred_features
                # relevant masks
                x["tmask"] = tm
                x["pmask"] = pm
                x["jmask"] = jm

                x["flow"] = flow_features

            else:
                x["table"] = table_features
                x["join"] = join_features
                x["pred"] = pred_features
                x["flow"] = flow_features

            X.append(x)
            Y.append(self.featurizer.normalize_val(true_val, total))
            cur_info = {}
            cur_info["num_tables"] = len(nodes)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["query_idx"] = query_idx

            # FIXME:
            # cur_info["total"] = total
            cur_info["total"] = 0.00
            sample_info.append(cur_info)

        assert len(Y) == len(sample_info) == len(X)
        return X,Y,sample_info

    def _get_feature_vectors(self, samples):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.
        '''
        start = time.time()

        if self.featurizer.featurization_type == "combined":
            X = []
        elif self.featurizer.featurization_type == "set":
            # X = defaultdict(list)
            X = []

        Y = []
        sample_info = []
        qidx = 0

        for i, qrep in enumerate(samples):
            if self.featurizer.featurization_type == "set":
                x,y,cur_info = self._get_query_features_set(qrep, qidx, i)
            elif self.featurizer.featurization_type == "combined":
                x,y,cur_info = self._get_query_features(qrep, qidx, i)
            else:
                assert False

            qidx += len(y)
            if self.featurizer.featurization_type == "combined":
                X += x
            else:
                X += x
                # for k,v in x.items():
                    # X[k] += v

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

            # if self.feattype == "combined":
                # return self.X[index], self.Y[index], self.info[index]
            # elif self.feattype == "set":
                # return self.X[index], self.Y[index], self.info[index]

