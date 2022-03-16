import time
import numpy as np
import pdb
import math
import pandas as pd
import json
import sys
import torch
from collections import defaultdict
import random

from query_representation.utils import *

from evaluation.eval_fns import *
from .dataset import QueryDataset, pad_sets, to_variable,\
        mscn_collate_fn,mscn_collate_fn_together

from .nets import *
from evaluation.flow_loss import FlowLoss, \
        get_optimization_variables, get_subsetg_vectors

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.optim.swa_utils import AveragedModel, SWALR

import wandb
import random

QERR_MIN_EPS=0.0
def qloss_torch(yhat, ytrue):
    assert yhat.shape == ytrue.shape

    epsilons = to_variable([QERR_MIN_EPS]*len(yhat)).float()

    ytrue = torch.max(ytrue, epsilons)
    yhat = torch.max(yhat, epsilons)

    errors = torch.max( (ytrue / yhat), (yhat / ytrue))
    return errors

def mse_pos(yhat, ytrue):
    assert yhat.shape == ytrue.shape
    errors = torch.nn.MSELoss(reduction="none")(yhat, ytrue)

    for i,err in enumerate(errors):
        if yhat[i] < ytrue[i]:
            errors[i] *= 10

    return errors

def mse_ranknet(yhat, ytrue):
    mseloss = torch.nn.MSELoss(reduction="mean")(yhat, ytrue)
    rloss = ranknet_loss(yhat, ytrue)
    return mseloss + 0.1*rloss

def ranknet_loss(batch_pred, batch_label):
    '''
    :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
    :param batch_label:  [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
    :return:
    '''
    batch_pred = batch_pred.unsqueeze(0)
    batch_label = batch_label.unsqueeze(0)
    # batch_pred = batch_pred.T
    # batch_label = batch_label.T
    sigma = 1.0

    batch_s_ij = torch.unsqueeze(batch_pred, dim=2) - torch.unsqueeze(batch_pred, dim=1)  # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j

    batch_p_ij = 1.0 / (torch.exp(-sigma * batch_s_ij) + 1.0)

    batch_std_diffs = torch.unsqueeze(batch_label, dim=2) - torch.unsqueeze(batch_label, dim=1)  # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
    batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

    # about reduction, both mean & sum would work, mean seems straightforward due to the fact that the number of pairs differs from query to query
    batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1), target=torch.triu(batch_std_p_ij, diagonal=1), reduction='mean')

    return batch_loss

class CardinalityEstimationAlg():

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        pass

    def train(self, training_samples, **kwargs):
        pass

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subplan). Each key should be ' ' separated
        list of aliases / table names
        '''
        pass

    def get_exp_name(self):
        name = self.__str__()
        if not hasattr(self, "rand_id"):
            self.rand_id = str(random.getrandbits(32))
            print("Experiment name will be: ", name + self.rand_id)

        name += self.rand_id
        return name

    def num_parameters(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        return 0

    def __str__(self):
        return self.__class__.__name__

    def save_model(self, save_dir="./", suffix_name=""):
        pass

def format_model_test_output(pred, samples, featurizer):
    all_ests = []
    query_idx = 0
    for sample in samples:
        ests = {}
        node_keys = list(sample["subset_graph"].nodes())
        if SOURCE_NODE in node_keys:
            node_keys.remove(SOURCE_NODE)
        node_keys.sort()
        for subq_idx, node in enumerate(node_keys):
            cards = sample["subset_graph"].nodes()[node]["cardinality"]
            alias_key = node
            idx = query_idx + subq_idx
            est_card = featurizer.unnormalize(pred[idx], cards["total"])
            assert est_card > 0
            ests[alias_key] = est_card

        all_ests.append(ests)
        query_idx += len(node_keys)
    return all_ests

class NN(CardinalityEstimationAlg):

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        for k, val in kwargs.items():
            self.__setattr__(k, val)

        # when estimates are log-normalized, then optimizing for mse is
        # basically equivalent to optimizing for q-error
        self.num_workers = 8
        if self.loss_func_name == "qloss":
            self.loss_func = qloss_torch
            self.load_query_together = False
        elif self.loss_func_name == "mse":
            self.loss_func = torch.nn.MSELoss(reduction="none")
            self.load_query_together = False
        elif self.loss_func_name == "mse_pos":
            self.loss_func = mse_pos
            self.load_query_together = False
        elif self.loss_func_name == "flowloss":
            self.loss_func = FlowLoss.apply
            self.load_query_together = True
            if self.mb_size > 16:
                self.mb_size = 1
            self.num_workers = 1
            # self.collate_fn = None
        elif self.loss_func_name == "mse+ranknet":
            self.loss_func = mse_ranknet
            self.load_query_together = True
            if self.mb_size > 16:
                self.mb_size = 1
        else:
            assert False

        if self.load_query_together:
            self.collate_fn = mscn_collate_fn_together
        else:
            if hasattr(self, "load_padded_mscn_feats"):
                if self.load_padded_mscn_feats:
                    self.collate_fn = None
                else:
                    self.collate_fn = mscn_collate_fn
            else:
                self.collate_fn = None

        self.eval_fn_handles = []
        for efn in self.eval_fns.split(","):
            self.eval_fn_handles.append(get_eval_fn(efn))

    def init_net(self, sample):
        net = self._init_net(sample)
        print(net)

        if self.optimizer_name == "ams":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=True, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=False, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=False, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(net.parameters(),
                    lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            assert False

        # if self.use_wandb:
            # wandb.watch(net)

        return net, optimizer

    def periodic_eval(self):
        if not self.use_wandb:
            return
        start = time.time()
        curerrs = {}
        for st, ds in self.eval_ds.items():
            samples = self.samples[st]
            preds = self._eval_ds(ds, samples)
            preds = format_model_test_output(preds,
                    samples, self.featurizer)

            # do evaluations
            for efunc in self.eval_fn_handles:
                if "Constraint" in str(efunc):
                    continue
                errors = efunc.eval(samples, preds,
                        args=None, samples_type=st,
                        result_dir=None,
                        user = self.featurizer.user,
                        db_name = self.featurizer.db_name,
                        db_host = self.featurizer.db_host,
                        port = self.featurizer.port,
                        num_processes = 16,
                        alg_name = self.__str__(),
                        save_pdf_plans=False,
                        use_wandb=False)

                err = np.mean(errors)
                wandb.log({str(efunc)+"-"+st: err, "epoch":self.epoch})
                curerrs[str(efunc)+"-"+st] = round(err,4)

        print("Epoch ", self.epoch, curerrs)
        print("periodic_eval took: ", time.time()-start)

    def update_flow_training_info(self):
        fstart = time.time()
        # precompute a whole bunch of training things
        self.flow_training_info = []
        # farchive = klepto.archives.dir_archive("./flow_info_archive",
                # cached=True, serialized=True)
        # farchive.load()
        new_seen = False
        for sample in self.training_samples:
            qkey = deterministic_hash(sample["sql"])
            # if qkey in farchive:
            if False:
                subsetg_vectors = farchive[qkey]
                assert len(subsetg_vectors) == 10
            else:
                new_seen = True
                subsetg_vectors = list(get_subsetg_vectors(sample,
                    self.cost_model))

            true_cards = np.zeros(len(subsetg_vectors[0]),
                    dtype=np.float32)
            nodes = list(sample["subset_graph"].nodes())

            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)

            nodes.sort()
            for i, node in enumerate(nodes):
                true_cards[i] = \
                    sample["subset_graph"].nodes()[node]["cardinality"]["actual"]

            trueC_vec, dgdxT, G, Q = \
                get_optimization_variables(true_cards,
                    subsetg_vectors[0], self.featurizer.min_val,
                        self.featurizer.max_val,
                        self.featurizer.ynormalization,
                        subsetg_vectors[4],
                        subsetg_vectors[5],
                        subsetg_vectors[3],
                        subsetg_vectors[1],
                        subsetg_vectors[2],
                        subsetg_vectors[6],
                        subsetg_vectors[7],
                        self.cost_model, subsetg_vectors[-1])

            Gv = to_variable(np.zeros(len(subsetg_vectors[0]))).float()
            Gv[subsetg_vectors[-2]] = 1.0
            trueC_vec = to_variable(trueC_vec).float()
            dgdxT = to_variable(dgdxT).float()
            G = to_variable(G).float()
            Q = to_variable(Q).float()

            trueC = torch.eye(len(trueC_vec)).float().detach()
            for i, curC in enumerate(trueC_vec):
                trueC[i,i] = curC

            invG = torch.inverse(G)
            v = invG @ Gv
            left = (Gv @ torch.transpose(invG,0,1)) @ torch.transpose(Q, 0, 1)
            right = Q @ (v)
            left = left.detach().cpu()
            right = right.detach().cpu()
            opt_flow_loss = left @ trueC @ right
            del trueC

            # print(opt_flow_loss)
            # pdb.set_trace()

            self.flow_training_info.append((subsetg_vectors, trueC_vec,
                    opt_flow_loss))

        print("precomputing flow info took: ", time.time()-fstart)

    def train(self, training_samples, **kwargs):
        assert isinstance(training_samples[0], dict)
        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples

        self.seen_subplans = set()
        for sample in training_samples:
            for node in sample["subset_graph"].nodes():
                self.seen_subplans.add(str(node))

        self.trainds = self.init_dataset(training_samples,
                self.load_query_together)
        self.trainloader = data.DataLoader(self.trainds,
                batch_size=self.mb_size, shuffle=True,
                collate_fn=self.collate_fn,
                # num_workers=self.num_workers
                )

        self.eval_ds = {}
        self.samples = {}
        if self.eval_epoch < self.max_epochs:
            # create eval loaders
            # self.eval_ds["train"] = self.trainds
            # self.samples["train"] = training_samples

            if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                self.eval_ds["val"] = self.init_dataset(kwargs["valqs"], False)
                self.samples["val"] = kwargs["valqs"]

            if "testqs" in kwargs and len(kwargs["testqs"]) > 0:
                self.eval_ds["test"] = self.init_dataset(kwargs["testqs"],
                        False)
                self.samples["test"] = kwargs["testqs"]

        # TODO: initialize self.num_features
        self.net, self.optimizer = self.init_net(self.trainds[0])

        model_size = self.num_parameters()
        print("""Training samples: {}, Model size: {}""".
                format(len(self.trainds), model_size))

        if "flow" in self.loss_func_name:
            self.update_flow_training_info()

        if self.training_opt == "swa":
            self.swa_net = AveragedModel(self.net)
            # self.swa_start = self.swa_start
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.opt_lr)

        for self.epoch in range(0,self.max_epochs):

            if self.epoch % self.eval_epoch == 0 \
                    and self.epoch != 0:
                self.periodic_eval()

            self.train_one_epoch()

        if self.training_opt == "swa":
            torch.optim.swa_utils.update_bn(self.trainloader, self.swa_net)

    def _eval_ds(self, ds, samples=None):
        torch.set_grad_enabled(False)

        if self.training_opt == "swa":
            net = self.swa_net
        else:
            net = self.net

        # important to not shuffle the data so correct order preserved!
        loader = data.DataLoader(ds,
                batch_size=5000, shuffle=False,
                collate_fn = None
                # collate_fn=self.collate_fn
                )

        allpreds = []

        for (xbatch,ybatch,info) in loader:
            ybatch = ybatch.to(device, non_blocking=True)

            if self.mask_unseen_subplans:
                start = time.time()
                pf_mask = torch.from_numpy(self.featurizer.pred_onehot_mask).float()
                jf_mask = torch.from_numpy(self.featurizer.join_onehot_mask).float()
                tf_mask = torch.from_numpy(self.featurizer.table_onehot_mask).float()

                for ci,curnode in enumerate(info["node"]):
                    if not curnode in self.seen_subplans:
                        if self.featurizer.pred_features:
                            xbatch["pred"][ci] = xbatch["pred"][ci] * pf_mask
                        if self.featurizer.join_features:
                            xbatch["join"][ci] = xbatch["join"][ci] * jf_mask
                        if self.featurizer.table_features:
                            xbatch["table"][ci] = xbatch["table"][ci] * tf_mask

                # print("masking unseen subplans took: ", time.time()-start)


            if self.subplan_level_outputs:
                pred = net(xbatch).squeeze(1)
                idxs = torch.zeros(pred.shape,dtype=torch.bool)
                for i, nt in enumerate(info["num_tables"]):
                    if nt >= 10:
                        nt = 10
                    nt -= 1
                    idxs[i,nt] = True
                pred = pred[idxs]
            else:
                pred = net(xbatch).squeeze(1)

            allpreds.append(pred)

        preds = torch.cat(allpreds).detach().cpu().numpy()
        torch.set_grad_enabled(True)

        if self.heuristic_unseen_preds == "pg" and samples is not None:
            newpreds = []
            query_idx = 0
            for sample in samples:
                node_keys = list(sample["subset_graph"].nodes())
                if SOURCE_NODE in node_keys:
                    node_keys.remove(SOURCE_NODE)
                node_keys.sort()
                for subq_idx, node in enumerate(node_keys):
                    cards = sample["subset_graph"].nodes()[node]["cardinality"]
                    idx = query_idx + subq_idx
                    est_card = preds[idx]
                    # were all columns in this subplan + constants seen in the
                    # training set?
                    print(node)

                    pdb.set_trace()

            preds = np.array(newpreds)
            pdb.set_trace()

        return preds

    def _get_onehot_mask(self, vec):
        tmask = ~np.array(vec, dtype="bool")
        ptrue = self.onehot_mask_truep
        pfalse = 1-self.onehot_mask_truep

        # probabilities are switched
        bools = np.random.choice(a=[False, True], size=(len(tmask),),
                p=[ptrue,pfalse])
        tmask *= bools
        tmask = ~tmask
        tmask = torch.from_numpy(tmask).float()
        return tmask

    def _get_onehot_mask2(self, xbatch):

        if self.onehot_dropout == 2:
            # doesn't depend on xbatch at all
            tf_mask = self._get_onehot_mask(self.featurizer.table_onehot_mask)
            jf_mask = self._get_onehot_mask(self.featurizer.join_onehot_mask)
            pf_mask = self._get_onehot_mask(self.featurizer.pred_onehot_mask)
            return tf_mask, jf_mask, pf_mask

        elif self.onehot_dropout == 3:
            # don't distinguish between onehot vs stats features, just apply
            # dropout everywhere
            ptrue = self.onehot_mask_truep
            pfalse = 1-self.onehot_mask_truep
            tf_mask = np.random.choice(a=[True, False],
                    size=(xbatch["table"].shape[0], len(self.featurizer.table_onehot_mask),),
                    p=[ptrue,pfalse])
            tf_mask = tf_mask[:,None,:]
            jf_mask = np.random.choice(a=[True, False],
                    size=(xbatch["join"].shape[0], len(self.featurizer.join_onehot_mask),),
                    p=[ptrue,pfalse])
            jf_mask = jf_mask[:,None,:]
            pf_mask = np.random.choice(a=[True, False],
                    size=(xbatch["pred"].shape[0], len(self.featurizer.pred_onehot_mask),),
                    p=[ptrue,pfalse])
            pf_mask = pf_mask[:,None,:]
        elif self.onehot_dropout == 4:
            # note: we need to do extra work here because we don't want the
            # stats features to be affected by the onehot mask
            ptrue = self.onehot_mask_truep
            pfalse = 1-self.onehot_mask_truep
            tf_mask = np.random.choice(a=[False, True],
                    size=(xbatch["table"].shape[0], len(self.featurizer.table_onehot_mask),),
                    p=[ptrue,pfalse])
            # negating, turns stats features to zero, and onehot features to
            # one
            tmask = ~np.array(self.featurizer.table_onehot_mask, dtype="bool")
            tf_mask *= tmask
            # at this point, the stats features would have become zero; now we
            # can negate it, and we'll get a dropout on only the onehot
            # features
            tf_mask = ~tf_mask
            # fixing dimensions to match xbatch
            tf_mask = tf_mask[:,None,:]

            ## repeating process with join features and pred features
            jf_mask = np.random.choice(a=[False, True],
                    size=(xbatch["join"].shape[0], len(self.featurizer.join_onehot_mask),),
                    p=[ptrue,pfalse])
            jmask = ~np.array(self.featurizer.join_onehot_mask, dtype="bool")
            jf_mask *= jmask
            jf_mask = ~jf_mask
            jf_mask = jf_mask[:,None,:]

            pf_mask = np.random.choice(a=[False, True],
                    size=(xbatch["pred"].shape[0], len(self.featurizer.pred_onehot_mask),),
                    p=[ptrue,pfalse])
            pmask = ~np.array(self.featurizer.pred_onehot_mask, dtype="bool")
            pf_mask *= pmask
            pf_mask = ~pf_mask
            pf_mask = pf_mask[:,None,:]
        else:
            assert False

        tf_mask = torch.from_numpy(tf_mask).float()
        jf_mask = torch.from_numpy(jf_mask).float()
        pf_mask = torch.from_numpy(pf_mask).float()
        return tf_mask, jf_mask, pf_mask

    def train_one_epoch(self):
        if self.loss_func_name == "flowloss":
            torch.set_num_threads(1)

        start = time.time()
        backtimes = []
        ftimes = []
        epoch_losses = []

        for idx, (xbatch, ybatch,info) in enumerate(self.trainloader):
            # TODO: load_query_together things
            ybatch = ybatch.to(device, non_blocking=True)

            if self.onehot_dropout == 0:
                pass

            elif self.onehot_dropout == 1:
                if random.random() < 0.5:
                    # want to change the inputs by selectively zero-ing out
                    # some things
                    pf_mask = torch.from_numpy(self.featurizer.pred_onehot_mask).float()
                    jf_mask = torch.from_numpy(self.featurizer.join_onehot_mask).float()
                    tf_mask = torch.from_numpy(self.featurizer.table_onehot_mask).float()

                    if self.featurizer.pred_features:
                        xbatch["pred"] = xbatch["pred"] * pf_mask
                    if self.featurizer.join_features:
                        xbatch["join"] = xbatch["join"] * jf_mask
                    if self.featurizer.table_features:
                        xbatch["table"] = xbatch["table"] * tf_mask

            elif self.onehot_dropout == 2:
                tf_mask = self._get_onehot_mask(self.featurizer.table_onehot_mask)
                jf_mask = self._get_onehot_mask(self.featurizer.join_onehot_mask)
                pf_mask = self._get_onehot_mask(self.featurizer.pred_onehot_mask)

                if self.featurizer.pred_features:
                    xbatch["pred"] = xbatch["pred"] * pf_mask
                if self.featurizer.join_features:
                    xbatch["join"] = xbatch["join"] * jf_mask
                if self.featurizer.table_features:
                    xbatch["table"] = xbatch["table"] * tf_mask
            else:
                # tf_mask = self._get_onehot_mask(self.featurizer.table_onehot_mask)
                # jf_mask = self._get_onehot_mask(self.featurizer.join_onehot_mask)
                # pf_mask = self._get_onehot_mask(self.featurizer.pred_onehot_mask)
                tf_mask, jf_mask, pf_mask = self._get_onehot_mask2(xbatch)

                if self.featurizer.pred_features:
                    xbatch["pred"] = xbatch["pred"] * pf_mask
                if self.featurizer.join_features:
                    xbatch["join"] = xbatch["join"] * jf_mask
                if self.featurizer.table_features:
                    xbatch["table"] = xbatch["table"] * tf_mask

            # elif self.onehot_dropout == 3:
                # print(xbatch["table"].shape)
                # num_batches = xbatch["table"].shape[0]
                # tf_mask = self._get_onehot_mask_per_subplan(
                        # num_batches, xbatch["table"].shape[1],
                        # self.featurizer.table_onehot_mask)
                # jf_mask = self._get_onehot_mask_per_subplan(num_batches,
                        # xbatch["join"].shape[1],
                        # self.featurizer.join_onehot_mask)
                # pf_mask = self._get_onehot_mask_per_subplan(num_batches,
                        # xbatch["pred"].shape[1],
                        # self.featurizer.pred_onehot_mask)
                # print(tf_mask)
                # print(tf_mask.shape)
                # pdb.set_trace()

            if self.subplan_level_outputs:
                pred = self.net(xbatch).squeeze(1)
                idxs = torch.zeros(pred.shape,dtype=torch.bool)
                for i, nt in enumerate(info["num_tables"]):
                    if nt >= 10:
                        nt = 10
                    nt -= 1
                    idxs[i,nt] = True
                pred = pred[idxs]
            else:
                pred = self.net(xbatch).squeeze(1)

            assert pred.shape == ybatch.shape

            if self.loss_func_name == "flowloss":
                assert self.load_query_together
                qstart = 0
                losses = []

                # print(self.mb_size)
                # print(len(info))
                # pdb.set_trace()

                for cur_info in info:
                    if "query_idx" not in cur_info[0]:
                        print(cur_info)
                        pdb.set_trace()
                    qidx = cur_info[0]["query_idx"]
                    assert qidx == cur_info[1]["query_idx"]
                    subsetg_vectors, trueC_vec, opt_loss = \
                            self.flow_training_info[qidx]

                    assert len(subsetg_vectors) == 10
                    fstart = time.time()

                    cur_loss = self.loss_func(
                            pred[qstart:qstart+len(cur_info)],
                            ybatch[qstart:qstart+len(cur_info)],
                            self.featurizer.ynormalization,
                            self.featurizer.min_val,
                            self.featurizer.max_val,
                            [(subsetg_vectors, trueC_vec, opt_loss)],
                            self.normalize_flow_loss,
                            None,
                            self.cost_model)
                    ftimes.append(time.time()-fstart)
                    losses.append(cur_loss)
                    qstart += len(cur_info)

                losses = torch.stack(losses)
                loss = losses.sum() / len(losses)
            else:
                losses = self.loss_func(pred, ybatch)
                if len(losses.shape) != 0:
                    loss = losses.sum() / len(losses)
                else:
                    loss = losses

            epoch_losses.append(loss.item())

            if self.onehot_reg:
                reg_loss = None
                for name, param in self.net.named_parameters():
                    if name == "sample_mlp1.weight":
                        mask = torch.from_numpy(~np.array(self.featurizer.table_onehot_mask,
                            dtype="bool")).float()
                    elif name == "join_mlp1.weight":
                        mask = torch.from_numpy(~np.array(self.featurizer.join_onehot_mask,
                            dtype="bool")).float()
                    elif name == "predicate_mlp1.weight":
                        mask = torch.from_numpy(~np.array(self.featurizer.pred_onehot_mask,
                            dtype="bool")).float()
                    else:
                        continue

                    reg_param = param*mask
                    if reg_loss is None:
                        reg_loss = reg_param.norm(p=2)
                    else:
                        reg_loss = reg_loss + reg_param.norm(p=2)

                if reg_loss is not None:
                    loss += self.onehot_reg_decay * reg_loss

            if self.training_opt == "swa":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.epoch > self.swa_start:
                    self.swa_net.update_parameters(self.net)
                    self.swa_scheduler.step()
            else:
                bstart = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                backtimes.append(time.time()-bstart)
                if self.clip_gradient is not None:
                    clip_grad_norm_(self.net.parameters(), self.clip_gradient)
                self.optimizer.step()

        curloss = round(float(sum(epoch_losses))/len(epoch_losses),6)
        print("Epoch {} took {}, Avg Loss: {}".format(self.epoch,
            round(time.time()-start, 2), curloss))
        print("Backward avg time: {}, Forward avg time: {}".format(\
                np.mean(backtimes), np.mean(ftimes)))

        if self.use_wandb:
            wandb.log({"TrainLoss": curloss, "epoch":self.epoch})

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subplan). Each key should be ' ' separated
        list of aliases / table names
        '''
        testds = self.init_dataset(test_samples, False)
        # testds = QueryDataset(test_samples, self.featurizer,
                # False,
                # load_padded_mscn_feats=self.load_padded_mscn_feats)

        preds = self._eval_ds(testds, test_samples)

        return format_model_test_output(preds, test_samples, self.featurizer)

    def get_exp_name(self):
        name = self.__str__()
        if not hasattr(self, "rand_id"):
            self.rand_id = str(random.getrandbits(32))
            print("Experiment name will be: ", name + self.rand_id)

        name += self.rand_id
        return name

    def num_parameters(self):
        def _calc_size(net):
            model_parameters = net.parameters()
            params = sum([np.prod(p.size()) for p in model_parameters])
            # convert to MB
            return params*4 / 1e6

        num_params = _calc_size(self.net)
        return num_params

    def __str__(self):
        return self.__class__.__name__

    def save_model(self, save_dir="./", suffix_name=""):
        pass


class SavedPreds(CardinalityEstimationAlg):
    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        self.model_dir = kwargs["model_dir"]
        self.max_epochs = 0

    def train(self, training_samples, **kwargs):
        assert os.path.exists(self.model_dir)
        self.saved_preds = load_object_gzip(self.model_dir + "/preds.pkl")

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subquery). Each key should be ' ' separated
        list of aliases / table names
        '''
        preds = []
        for sample in test_samples:
            assert sample["name"] in self.saved_preds
            preds.append(self.saved_preds[sample["name"]])
        return preds

    def get_exp_name(self):
        old_name = os.path.basename(self.model_dir)
        name = "SavedRun-" + old_name
        return name

    def num_parameters(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        return 0

    def __str__(self):
        return "SavedAlg"

    def save_model(self, save_dir="./", suffix_name=""):
        pass

class Postgres(CardinalityEstimationAlg):
    def test(self, test_samples, **kwargs):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            nodes = list(sample["subset_graph"].nodes())

            for alias_key in nodes:
                info = sample["subset_graph"].nodes()[alias_key]
                true_card = info["cardinality"]["actual"]
                if "expected" not in info["cardinality"]:
                    continue
                est = info["cardinality"]["expected"]
                pred_dict[(alias_key)] = est

            preds.append(pred_dict)
        return preds

    def get_exp_name(self):
        return self.__str__()

    def __str__(self):
        return "Postgres"

class TrueCardinalities(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            nodes = list(sample["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            for alias_key in nodes:
                info = sample["subset_graph"].nodes()[alias_key]
                pred_dict[(alias_key)] = info["cardinality"]["actual"]
            preds.append(pred_dict)
        return preds

    def get_exp_name(self):
        return self.__str__()

    def __str__(self):
        return "True"

class TrueRandom(CardinalityEstimationAlg):
    def __init__(self):
        # max percentage noise added / subtracted to true values
        self.max_noise = random.randint(1,500)

    def test(self, test_samples):
        # choose noise type
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                true_card = info["cardinality"]["actual"]
                # add noise
                noise_perc = random.randint(1,self.max_noise)
                noise = (true_card * noise_perc) / 100.00
                if random.random() % 2 == 0:
                    updated_card = true_card + noise
                else:
                    updated_card = true_card - noise
                if updated_card <= 0:
                    updated_card = 1
                pred_dict[(alias_key)] = updated_card
            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true_random"

class TrueRank(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            all_cards = []
            for alias_key, info in sample["subset_graph"].nodes().items():
                # pred_dict[(alias_key)] = info["cardinality"]["actual"]
                card = info["cardinality"]["actual"]
                exp = info["cardinality"]["expected"]
                all_cards.append([alias_key, card, exp])
            all_cards.sort(key = lambda x : x[1])

            for i, (alias_key, true_est, pgest) in enumerate(all_cards):
                if i == 0:
                    pred_dict[(alias_key)] = pgest
                    continue
                prev_est = all_cards[i-1][2]
                prev_alias = all_cards[i-1][0]
                if pgest >= prev_est:
                    pred_dict[(alias_key)] = pgest
                else:
                    updated_est = prev_est
                    # updated_est = prev_est + 1000
                    # updated_est = true_est
                    all_cards[i][2] = updated_est
                    pred_dict[(alias_key)] = updated_est

            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true_rank"

class TrueRankTables(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            all_cards_nt = defaultdict(list)
            for alias_key, info in sample["subset_graph"].nodes().items():
                # pred_dict[(alias_key)] = info["cardinality"]["actual"]
                card = info["cardinality"]["actual"]
                exp = info["cardinality"]["expected"]
                nt = len(alias_key)
                all_cards_nt[nt].append([alias_key,card,exp])

            for _,all_cards in all_cards_nt.items():
                all_cards.sort(key = lambda x : x[1])
                for i, (alias_key, true_est, pgest) in enumerate(all_cards):
                    if i == 0:
                        pred_dict[(alias_key)] = pgest
                        continue
                    prev_est = all_cards[i-1][2]
                    prev_alias = all_cards[i-1][0]
                    if pgest >= prev_est:
                        pred_dict[(alias_key)] = pgest
                    else:
                        updated_est = prev_est
                        # updated_est = prev_est + 1000
                        # updated_est = true_est
                        all_cards[i][2] = updated_est
                        pred_dict[(alias_key)] = updated_est

            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true_rank_tables"

class Random(CardinalityEstimationAlg):
    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                total = info["cardinality"]["total"]
                est = random.random()*total
                pred_dict[(alias_key)] = est
            preds.append(pred_dict)
        return preds

class XGBoost(CardinalityEstimationAlg):
    def __init__(self, **kwargs):
        for k, val in kwargs.items():
            self.__setattr__(k, val)

    def init_dataset(self, samples):
        ds = QueryDataset(samples, self.featurizer, False)
        X = ds.X.cpu().numpy()
        Y = ds.Y.cpu().numpy()
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        del(ds)
        return X, Y

    def load_model(self, model_dir):
        import xgboost as xgb
        model_path = model_dir + "/xgb_model.json"
        import xgboost as xgb
        self.xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
        self.xgb_model.load_model(model_path)
        print("*****loaded model*****")

    def train(self, training_samples, **kwargs):
        import xgboost as xgb
        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples

        X,Y = self.init_dataset(training_samples)

        if self.grid_search:
            parameters = {'learning_rate':(0.001, 0.01),
                    'n_estimators':(100, 250, 500, 1000),
                    'loss': ['ls'],
                    'max_depth':(3, 6, 8, 10),
                    'subsample':(1.0, 0.8, 0.5)}

            xgb_model = GradientBoostingRegressor()
            self.xgb_model = RandomizedSearchCV(xgb_model, parameters, n_jobs=-1,
                    verbose=1)
            self.xgb_model.fit(X, Y)
            print("*******************BEST ESTIMATOR FOUND**************")
            print(self.xgb_model.best_estimator_)
            print("*******************BEST ESTIMATOR DONE**************")
        else:
            import xgboost as xgb
            self.xgb_model = xgb.XGBRegressor(tree_method=self.tree_method,
                          objective="reg:squarederror",
                          verbosity=1,
                          scale_pos_weight=0,
                          learning_rate=self.lr,
                          colsample_bytree = 1.0,
                          subsample = self.subsample,
                          n_estimators=self.n_estimators,
                          reg_alpha = 0.0,
                          max_depth=self.max_depth,
                          gamma=0)
            self.xgb_model.fit(X,Y, verbose=1)

        if hasattr(self, "result_dir") and self.result_dir is not None:
            exp_name = self.get_exp_name()
            exp_dir = os.path.join(self.result_dir, exp_name)
            self.xgb_model.save_model(exp_dir + "/xgb_model.json")

    def test(self, test_samples):
        X,Y = self.init_dataset(test_samples)
        pred = self.xgb_model.predict(X)
        return format_model_test_output(pred, test_samples, self.featurizer)

    def __str__(self):
        return self.__class__.__name__

class RandomForest(CardinalityEstimationAlg):
    def __init__(self, **kwargs):
        for k, val in kwargs.items():
            self.__setattr__(k, val)

    def init_dataset(self, samples):
        ds = QueryDataset(samples, self.featurizer, False)
        X = ds.X.cpu().numpy()
        Y = ds.Y.cpu().numpy()
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        del(ds)
        return X, Y

    def load_model(self, model_dir):
        pass

    def train(self, training_samples, **kwargs):
        from sklearn.ensemble import RandomForestRegressor

        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples

        X,Y = self.init_dataset(training_samples)

        if self.grid_search:
            pass
        else:
            self.model = RandomForestRegressor(n_jobs=-1,
                    verbose=2,
                    n_estimators = self.n_estimators,
                    max_depth = self.max_depth)
            self.model.fit(X, Y)

    def test(self, test_samples):
        X,Y = self.init_dataset(test_samples)
        pred = self.model.predict(X)
        # FIXME: why can't we just use get_query_estimates here?
        return format_model_test_output(pred, test_samples, self.featurizer)

    def __str__(self):
        return self.__class__.__name__
