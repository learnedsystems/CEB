import time
import numpy as np
import pdb
import math
import pandas as pd
import json
import sys
import xgboost as xgb
import random
import torch
from collections import defaultdict

from query_representation.utils import *
from .dataset import QueryDataset, pad_sets, to_variable
from .nets import *
# from .fcnn import FCNN
# from .mscn import MSCN

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.ensemble import GradientBoostingRegressor

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
            true_card = cards["actual"]
            ests[alias_key] = est_card

        all_ests.append(ests)
        query_idx += len(node_keys)
    return all_ests

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
        model_path = model_dir + "/xgb_model.json"
        self.xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
        self.xgb_model.load_model(model_path)
        print("*****loaded model*****")

    def train(self, training_samples, **kwargs):
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
        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples

        X,Y = self.init_dataset(training_samples)

        if self.grid_search:
            pass
        else:
            self.model = RandomForestRegressor(n_jobs=-1, verbose=2, **params)
            self.model.fit(X, Y)

    def test(self, test_samples):
        X,Y = self.init_dataset(test_samples)
        pred = self.model.predict(X)
        # FIXME: why can't we just use get_query_estimates here?
        return format_model_test_output(pred, test_samples, self.featurizer)

    def __str__(self):
        return self.__class__.__name__
