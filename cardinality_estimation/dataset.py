import torch
from torch.utils import data
from torch.autograd import Variable
from collections import defaultdict
import numpy as np
import time
import copy
import multiprocessing as mp
import math
import pickle

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

def get_query_features(qrep, dataset_qidx,
        query_idx, featurizer, load_padded_mscn_feats):
    '''
    Parallelizable function.
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
        x,y = featurizer.get_subplan_features(qrep,
                node)

        if featurizer.featurization_type == "set" \
            and load_padded_mscn_feats:
            start = time.time()
            tf,pf,jf,tm,pm,jm = \
                pad_sets([x["table"]], [x["pred"]], [x["join"]],
                        featurizer.max_tables,
                        featurizer.max_preds,
                        featurizer.max_joins)
            x["table"] = tf
            x["join"] = jf
            x["pred"] = pf
            # relevant masks
            x["tmask"] = tm
            x["pmask"] = pm
            x["jmask"] = jm
            x["flow"] = to_variable(x["flow"], requires_grad=False).float()

        X.append(x)
        Y.append(y)

        cur_info = {}
        cur_info["num_tables"] = len(node)
        cur_info["dataset_idx"] = dataset_qidx + node_idx
        cur_info["query_idx"] = query_idx
        sample_info.append(cur_info)

    return X,Y,sample_info

def get_queries_features(samples, dataset_qidx,
        start_query_idx, featurizer, load_padded_mscn_feats):
    X = []
    Y = []
    sample_info = []

    qidx = dataset_qidx
    for i, qrep in enumerate(samples):
        x,y,cur_info = get_query_features(qrep, qidx, start_query_idx+i,
                featurizer, load_padded_mscn_feats)
        qidx += len(y)
        X += x
        Y += y
        sample_info += cur_info

    return X,Y,sample_info

def mscn_collate_fn_together(data):
    start = time.time()
    alldata = defaultdict(list)
    # print(len(data))

    for di in range(len(data)):
        for feats in data[di][0]:
            for k,v in feats.items():
                alldata[k].append(v)

    xdata = {}
    for k,v in alldata.items():
        # xdata[k] = torch.stack(v)
        if k == "flow":
            if len(v[0]) == 0:
                xdata[k] = v
            else:
                # print(type(v))
                # print(type(v[0]))
                # pdb.set_trace()
                # xdata[k] = to_variable(v, requires_grad=False).float()
                xdata[k] = torch.stack(v)
        else:
            xdata[k] = torch.stack(v)

    # ys = data[0][1]

    # for cinfo in data
    # infos = [data[0][2]]

    ys = [d[1] for d in data]
    ys = torch.cat(ys)
    infos = [d[2] for d in data]
    # print(xdata["table"].shape)
    # print(ys.shape)

    # pdb.set_trace()

    return xdata,ys,infos

def mscn_collate_fn(data):
    '''
    TODO: faster impl.
    '''
    start = time.time()
    alltabs = []
    allpreds = []
    alljoins = []

    flows = []
    ys = []
    infos = []

    maxtabs = 0
    maxpreds = 0
    maxjoins = 0

    for d in data:
        alltabs.append(d[0]["table"])
        if len(alltabs[-1]) > maxtabs:
            maxtabs = len(alltabs[-1])

        allpreds.append(d[0]["pred"])
        if len(allpreds[-1]) > maxpreds:
            maxpreds = len(allpreds[-1])

        alljoins.append(d[0]["join"])
        if len(alljoins[-1]) > maxjoins:
            maxjoins = len(alljoins[-1])

        flows.append(d[0]["flow"])
        ys.append(d[1])
        infos.append(d[2])

    tf,pf,jf,tm,pm,jm = pad_sets(alltabs, allpreds,
            alljoins, maxtabs,maxpreds,maxjoins)

    # print(flows)
    # pdb.set_trace()
    print(flows)
    # flows = to_variable(flows, requires_grad=False).float()
    flows = torch.stack(flows).float()

    ys = to_variable(ys, requires_grad=False).float()
    data = {}
    data["table"] = tf
    data["pred"] = pf
    data["join"] = jf
    data["flow"] = flows
    data["tmask"] = tm
    data["pmask"] = pm
    data["jmask"] = jm

    return data,ys,infos

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
        # print(len(table_features))
        pred_features = all_pred_features[i]
        join_features = all_join_features[i]

        pred_features, predicate_mask = _handle_set_padding(pred_features,
                maxpreds)
        table_features, table_mask = _handle_set_padding(table_features,
                maxtabs)
        join_features, join_mask = _handle_set_padding(join_features,
                maxjoins)

        # print(table_features.shape, table_mask.shape)
        # print(join_features.shape, join_mask.shape)
        # print("Tables: ")
        # print(table_features)
        # print(table_mask)

        # pdb.set_trace()
        # print("Joins: ")
        # print(join_features)
        # print(join_mask)
        # pdb.set_trace()

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
        if self.load_query_together:
            self.start_idxs, self.idx_lens = self._update_idxs(samples)

        self.load_padded_mscn_feats = load_padded_mscn_feats
        self.featurizer = featurizer

        self.save_mscn_feats = False
        if self.load_padded_mscn_feats:
            self.save_mscn_feats = True
            fkeys = list(dir(self.featurizer))
            fkeys.sort()
            attrs = ""
            for k in fkeys:
                attrvals = getattr(featurizer, k)
                if not hasattr(attrvals, "__len__") and \
                    "method" not in str(attrvals):
                    attrs += str(k) + str(attrvals) + ";"

            attrs += "padded"+str(self.load_padded_mscn_feats)
            self.feathash = deterministic_hash(attrs)
            self.featdir = "./mscn_features/" + str(self.feathash)
            if os.path.exists(self.featdir):
                print("features saved before")
            else:
                print("features NOT saved before")

            make_dir("./mscn_features")
            make_dir(self.featdir)

        # shorthands
        self.ckey = self.featurizer.ckey
        self.minv = self.featurizer.min_val
        self.maxv = self.featurizer.max_val
        self.feattype = self.featurizer.featurization_type

        # TODO: we may want to avoid this, and convert them on the fly. Just
        # keep some indexing information around.
        self.X, self.Y, self.info = self._get_feature_vectors(samples)
        # self.X, self.Y, self.info = self._get_feature_vectors_par(samples)

        if self.load_query_together:
            self.num_samples = len(samples)
        else:
            self.num_samples = len(self.X)

    def _update_idxs(self, samples):
        qidx = 0
        idx_starts = []
        idx_lens = []
        for i, qrep in enumerate(samples):
            # TODO: can also save these values and generate features when
            # needed, without wasting memory
            idx_starts.append(qidx)
            nodes = list(qrep["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            idx_lens.append(len(nodes))
            qidx += len(nodes)
        return idx_starts, idx_lens

    def _load_mscn_features(self, qfeat_fn):
        with open(qfeat_fn, "rb") as f:
            data = pickle.load(f)
        return data["x"], data["y"]

    def _save_mscn_features(self, x,y,qfeat_fn):
        data = {"x":x, "y":y}
        with open(qfeat_fn, "wb") as f:
            pickle.dump(data, f)

    def _get_sample_info(self, qrep, dataset_qidx, query_idx):
        sample_info = []
        node_names = list(qrep["subset_graph"].nodes())
        if SOURCE_NODE in node_names:
            node_names.remove(SOURCE_NODE)
        node_names.sort()

        for node_idx, node in enumerate(node_names):
            cur_info = {}
            cur_info["num_tables"] = len(node)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["query_idx"] = query_idx
            cur_info["node"] = str(node)
            sample_info.append(cur_info)

        return sample_info

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

        if self.featurizer.sample_bitmap:
            assert self.featurizer.bitmap_dir is not None
            bitmapfn = os.path.join(self.featurizer.bitmap_dir, qrep["name"])
            assert os.path.exists(bitmapfn)
            with open(bitmapfn, "rb") as handle:
                sbitmaps = pickle.load(handle)
        else:
            sbitmaps = None

        for node_idx, node in enumerate(node_names):
            x,y = self.featurizer.get_subplan_features(qrep,
                    node, bitmaps=sbitmaps)

            if self.featurizer.featurization_type == "set" \
                and self.load_padded_mscn_feats:
                start = time.time()
                tf,pf,jf,tm,pm,jm = \
                    pad_sets([x["table"]], [x["pred"]], [x["join"]],
                            self.featurizer.max_tables,
                            self.featurizer.max_preds,
                            self.featurizer.max_joins)
                x["table"] = tf
                x["join"] = jf
                x["pred"] = pf
                # relevant masks
                x["tmask"] = tm
                x["pmask"] = pm
                x["jmask"] = jm

            x["flow"] = to_variable(x["flow"], requires_grad=False).float()

            X.append(x)
            Y.append(y)

            cur_info = {}
            cur_info["num_tables"] = len(node)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["query_idx"] = query_idx
            cur_info["node"] = str(node)
            sample_info.append(cur_info)

        return X,Y,sample_info

    def _get_feature_vectors_par(self, samples):

        start = time.time()
        X = []
        Y = []
        sample_info = []


        nump = 16

        batchsize = 200
        outbatch = math.ceil(len(samples) / batchsize)

        dsqidx = 0
        pool = mp.Pool(nump)
        for i in range(outbatch):
            startidx = i*batchsize
            endidx = startidx+batchsize
            endidx = min(endidx, len(samples))
            print(startidx, endidx)
            qreps = samples[startidx:endidx]

            par_args = []
            for qi, qrep in enumerate(qreps):
                par_args.append((qrep, dsqidx, startidx+qi,
                                self.featurizer, self.load_padded_mscn_feats))
                dsqidx += len(qrep["subset_graph"].nodes())

            print("par args: ", len(par_args))

            # with mp.Pool(nump) as p:
            res = pool.starmap(get_query_features, par_args)
            for r in res:
                X += r[0]
                Y += r[1]
                sample_info += r[2]

        pool.close()
        pdb.set_trace()

        # par_args = []
        # batchsize = math.ceil(len(samples) / nump)

        # qidxstart = 0
        # for i in range(nump):
            # startidx = i*nump
            # end = startidx+batchsize
            # end = min(len(samples), end)
            # qreps = samples[startidx:end]

            # par_args.append((qreps, qidxstart, startidx, self.featurizer,
                # self.load_padded_mscn_feats))
            # for qr in qreps:
                # qidxstart += len(qr["subset_graph"].nodes())

        # # pool = mp.Pool(nump)
        # # res = pool.starmap(get_queries_features, par_args)
        # with mp.Pool(nump) as p:
            # res = p.starmap(get_queries_features, par_args)

            # # len(res) == 16; len(res[0]) == 3 # Xs,Ys,infos
            # for r in res:
                # X += r[0]
                # Y += r[1]
                # sample_info += r[2]

        # # pdb.set_trace()
        # pool.close()

        print("Extracting features took: ", time.time() - start)
        # TODO: handle this somehow
        if self.featurizer.featurization_type == "combined":
            X = to_variable(X, requires_grad=False).float()
        elif self.featurizer.featurization_type == "set":
            # don't need to do anything, since padding+masks is handled later
            pass

        Y = to_variable(Y, requires_grad=False).float()

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
            qhash = str(deterministic_hash(qrep["sql"]))

            if self.save_mscn_feats and \
                    "job" not in qrep["template_name"]:
                featfn = os.path.join(self.featdir, qhash) + ".pkl"
                if os.path.exists(featfn):
                    x,y = self._load_mscn_features(featfn)
                    cur_info = self._get_sample_info(qrep, qidx, i)
                else:
                    x,y,cur_info = self._get_query_features(qrep, qidx, i)
                    self._save_mscn_features(x,y,featfn)
            else:
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
            # assert False, "needs to be implemented"
            start_idx = self.start_idxs[index]
            end_idx = start_idx + self.idx_lens[index]
            # end_idx = start_idx + 2
            # print(start_idx, end_idx)
            # return self.X[start_idx], self.Y[start_idx], \
                    # self.info[start_idx]

            return self.X[start_idx:end_idx], self.Y[start_idx:end_idx], \
                    self.info[start_idx:end_idx]

            # if self.feattype == "combined":
                # return self.X[start_idx:end_idx], self.Y[start_idx:end_idx], \
                        # self.info[start_idx:end_idx]
            # elif self.feattype == "set":
                # print(start_idx, end_idx)
                # pdb.set_trace()
        else:
            return self.X[index], self.Y[index], self.info[index]
