import time
import numpy as np
import pdb
import math
import pandas as pd
import json
import sys
import random
import torch
from collections import defaultdict

from query_representation.utils import *

from evaluation.eval_fns import *
from .dataset import QueryDataset, pad_sets, to_variable,\
        mscn_collate_fn
from .nets import *
# from evaluation.flow_loss import FlowLoss, \
        # get_optimization_variables, get_subsetg_vectors

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_

import wandb

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
            est_card = featurizer.unnormalize(pred[idx])
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
        if self.loss_func_name == "qloss":
            self.loss_func = qloss_torch
            self.load_query_together = False
        elif self.loss_func_name == "mse":
            self.loss_func = torch.nn.MSELoss(reduction="none")
            self.load_query_together = False
        elif self.loss_func_name == "flowloss":
            self.loss = FlowLoss.apply
            self.load_query_together = True
        else:
            assert False

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

        if self.use_wandb:
            wandb.watch(net)

        return net, optimizer

    def periodic_eval(self):
        if not self.use_wandb:
            return
        start = time.time()
        for st, ds in self.eval_ds.items():
            samples = self.samples[st]
            preds = self._eval_ds(ds, samples)
            preds = format_model_test_output(preds,
                    samples, self.featurizer)

            # do evaluations
            for efunc in self.eval_fn_handles:
                errors = efunc.eval(samples, preds,
                        args=None, samples_type=st,
                        result_dir=None,
                        user = self.featurizer.user,
                        db_name = self.featurizer.db_name,
                        db_host = self.featurizer.db_host,
                        port = self.featurizer.port,
                        num_processes = -1,
                        alg_name = self.__str__(),
                        save_pdf_plans=False,
                        use_wandb=False)

                err = np.mean(errors)
                wandb.log({str(efunc)+"-"+st: err, "epoch":self.epoch})

        print("periodic_eval took: ", time.time()-start)

    def update_flow_training_info(self):
        print("precomputing flow loss info")
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

            self.flow_training_info.append((subsetg_vectors, trueC_vec,
                    opt_flow_loss))

        print("precomputing flow info took: ", time.time()-fstart)

    def train(self, training_samples, **kwargs):
        assert isinstance(training_samples[0], dict)
        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples

        self.trainds = self.init_dataset(training_samples)
        self.trainloader = data.DataLoader(self.trainds,
                batch_size=self.mb_size, shuffle=True,
                collate_fn=self.collate_fn,
                num_workers=8)

        self.eval_ds = {}
        self.samples = {}
        if self.eval_epoch < self.max_epochs:
            # create eval loaders
            self.eval_ds["train"] = self.trainds
            self.samples["train"] = training_samples
            if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                self.eval_ds["val"] = self.init_dataset(kwargs["valqs"])
                self.samples["val"] = kwargs["valqs"]

            if "testqs" in kwargs and len(kwargs["testqs"]) > 0:
                self.eval_ds["test"] = self.init_dataset(kwargs["testqs"])
                self.samples["test"] = kwargs["testqs"]

        # TODO: initialize self.num_features
        self.net, self.optimizer = self.init_net(self.trainds[0])

        model_size = self.num_parameters()
        print("""Training samples: {}, Model size: {}""".
                format(len(self.trainds), model_size))

        if "flow" in self.loss_func_name:
            self.update_flow_training_info()

        for self.epoch in range(0,self.max_epochs):
            if self.epoch % self.eval_epoch == 0:
                self.periodic_eval()

            self.train_one_epoch()

    def _eval_ds(self, ds, samples=None):
        torch.set_grad_enabled(False)
        # important to not shuffle the data so correct order preserved!
        loader = data.DataLoader(ds,
                batch_size=5000, shuffle=False,
                collate_fn=self.collate_fn)

        allpreds = []

        for (xbatch,ybatch,info) in loader:
            ybatch = ybatch.to(device, non_blocking=True)
            pred = self.net(xbatch).squeeze(1)
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

    def train_one_epoch(self):
        start = time.time()
        epoch_losses = []
        for idx, (xbatch, ybatch,info) in enumerate(self.trainloader):
            ybatch = ybatch.to(device, non_blocking=True)
            pred = self.net(xbatch).squeeze(1)
            assert pred.shape == ybatch.shape

            losses = self.loss_func(pred, ybatch)
            loss = losses.sum() / len(losses)

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(self.net.parameters(), self.clip_gradient)
            self.optimizer.step()
            epoch_losses.append(loss.item())

        curloss = round(float(sum(epoch_losses))/len(epoch_losses),6)
        print("Epoch {} took {}, Avg Loss: {}".format(self.epoch,
            round(time.time()-start, 2), curloss))

        if self.use_wandb:
            wandb.log({"TrainLoss": curloss, "epoch":self.epoch})

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subplan). Each key should be ' ' separated
        list of aliases / table names
        '''
        testds = self.init_dataset(test_samples)
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
