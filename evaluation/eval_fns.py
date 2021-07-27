import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plan_losses import PPC, PlanCost
from query_representation.utils import deterministic_hash,make_dir
from query_representation.viz import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import multiprocessing as mp
import random
from collections import defaultdict
import pandas as pd
import networkx as nx
import os

import pdb

def get_eval_fn(loss_name):
    if loss_name == "qerr":
        return QError()
    elif loss_name == "mse":
        return MSE()
    elif loss_name == "abs":
        return AbsError()
    elif loss_name == "ppc":
        return PostgresPlanCost()
    elif loss_name == "plancost":
        return SimplePlanCost()
    elif loss_name == "flowloss":
        return FlowLoss()
    else:
        assert False

class EvalFunc():
    def __init__(self, **kwargs):
        pass

    def eval(self, qreps, preds, **kwargs):
        '''
        @qreps: [qrep_1, ...qrep_N]
        @preds: [{},...,{}]

        @ret: [qerror_1, ..., qerror_{num_subplans}]
        Each query has multiple subplans; the returned list flattens it into a
        single array. The subplans of a query are sorted alphabetically (see
        _get_all_cardinalities)
        '''
        pass

    def __str__(self):
        return self.__class__.__name__

    # TODO: stuff for saving logs

def fix_query(query):
    # these conditions were needed due to some edge cases while generating the
    # queries on the movie_info_idx table, but crashes pyscopg2 somewhere.
    # Removing them shouldn't effect the queries.
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in query:
        query = query.replace(bad_str1, "")
    if bad_str2 in query:
        query = query.replace(bad_str2, "")

    return query

def _get_all_cardinalities(qreps, preds):
    ytrue = []
    yhat = []
    for i, pred_subsets in enumerate(preds):
        qrep = qreps[i]["subset_graph"].nodes()
        keys = list(pred_subsets.keys())
        keys.sort()
        for alias in keys:
            pred = pred_subsets[alias]
            actual = qrep[alias]["cardinality"]["actual"]
            if actual == 0:
                actual += 1
            ytrue.append(float(actual))
            yhat.append(float(pred))
    return np.array(ytrue), np.array(yhat)

class QError(EvalFunc):
    def eval(self, qreps, preds, **kwargs):
        '''
        '''
        assert len(preds) == len(qreps)
        assert isinstance(preds[0], dict)

        ytrue, yhat = _get_all_cardinalities(qreps, preds)
        assert len(ytrue) == len(yhat)
        assert 0.00 not in ytrue
        assert 0.00 not in yhat

        errors = np.maximum((ytrue / yhat), (yhat / ytrue))

        return errors

class AbsError(EvalFunc):
    def eval(self, qreps, preds, **kwargs):
        '''
        '''
        assert len(preds) == len(qreps)
        assert isinstance(preds[0], dict)

        ytrue, yhat = _get_all_cardinalities(qreps, preds)
        errors = np.abs(yhat - ytrue)
        return errors

class RelativeError(EvalFunc):
    def eval(self, qreps, preds, **kwargs):
        '''
        '''
        assert len(preds) == len(qreps)
        assert isinstance(preds[0], dict)
        ytrue, yhat = _get_all_cardinalities(queries, preds)
        # TODO: may want to choose a minimum estimate
        # epsilons = np.array([1]*len(yhat))
        # ytrue = np.maximum(ytrue, epsilons)

        errors = np.abs(ytrue - yhat) / ytrue
        return errors

class PostgresPlanCost(EvalFunc):
    def eval(self, qreps, preds, user="arthurfleck",pwd="password",
            db_name="imdb", db_host="localhost", port=5432, num_processes=-1,
            result_dir=None, cost_model="cm1", **kwargs):
        ''''
        @kwargs:
            cost_model: this is just a convenient key to specify the PostgreSQL
            configuration to use. You can implement new versions in the function
            set_cost_model. e.g., cm1: disable materialization and parallelism, and
            enable all other flags.
        @ret:
            pg_costs:
            pg_plans:
            pg_sqls: TODO.
            TODO: decide how to save result logs, incl. sqls to execute.
        '''
        assert isinstance(qreps, list)
        assert isinstance(preds, list)
        assert isinstance(qreps[0], dict)
        if "alg_name" in kwargs:
            alg_name = kwargs["alg_name"]
        else:
            alg_name = "Est"

        if "LCARD_USER" in os.environ:
            user = os.environ["LCARD_USER"]
        if "LCARD_PORT" in os.environ:
            port = os.environ["LCARD_PORT"]

        if num_processes == -1:
            pool = mp.Pool(int(mp.cpu_count()))
        else:
            pool = mp.Pool(num_processes)

        ppc = PPC(cost_model, user, pwd, db_host,
                port, db_name)

        est_cardinalities = []
        true_cardinalities = []
        sqls = []
        join_graphs = []

        for i, qrep in enumerate(qreps):
            sqls.append(qrep["sql"])
            join_graphs.append(qrep["join_graph"])
            ests = {}
            trues = {}
            predq = preds[i]
            for node, node_info in qrep["subset_graph"].nodes().items():
                est_card = predq[node]
                alias_key = ' '.join(node)
                trues[alias_key] = node_info["cardinality"]["actual"]
                if est_card == 0:
                    est_card += 1
                ests[alias_key] = est_card
            est_cardinalities.append(ests)
            true_cardinalities.append(trues)

        # some edge cases to handle to get the qreps to work in the PostgreSQL
        for i,sql in enumerate(sqls):
            sqls[i] = fix_query(sql)

        costs, opt_costs, plans, sqls = \
                    ppc.compute_costs(sqls, join_graphs,
                            true_cardinalities, est_cardinalities,
                            num_processes=num_processes,
                            pool=pool)

        if result_dir is not None:
            make_dir(result_dir)
            costs_fn = result_dir + "/" + cost_model + "_ppc.csv"
            if os.path.exists(costs_fn):
                costs_df = pd.read_csv(costs_fn)
            else:
                columns = ["sql_key", "plan", "exec_sql", "cost"]
                costs_df = pd.DataFrame(columns=columns)

            cur_costs = defaultdict(list)

            for i, qrep in enumerate(qreps):
                sql_key = str(deterministic_hash(qrep["sql"]))
                cur_costs["sql_key"].append(sql_key)
                cur_costs["plan"].append(plans[i])
                cur_costs["exec_sql"].append(sqls[i])
                cur_costs["cost"].append(costs[i])

            cur_df = pd.DataFrame(cur_costs)
            combined_df = pd.concat([costs_df, cur_df], ignore_index=True)
            combined_df.to_csv(costs_fn, index=False)

            pdf = PdfPages(os.path.join(result_dir, "query_plans.pdf"))
            for i, plan in enumerate(plans):
                if plan is None:
                    continue
                # no idea why explains we get from cursor.fetchall() have so
                # many indexes

                # we know cost of this; we know best cost;
                title_fmt = """{}. {} Estimates, Plan Cost: {}\n True cardinalities Plan Cost: {}"""
                title = title_fmt.format(qreps[i]["name"], alg_name, costs[i],
                        opt_costs[i])
                plot_explain_join_order(plan[0][0][0], true_cardinalities[i],
                        est_cardinalities[i], pdf, title)

            pdf.close()

        pool.close()
        return costs

class SimplePlanCost(EvalFunc):
    def eval(self, qreps, preds, cost_model="C",
            num_processes=-1, **kwargs):
        assert isinstance(qreps, list)
        assert isinstance(preds, list)
        assert isinstance(qreps[0], dict)

        if num_processes == -1:
            pool = mp.Pool(int(mp.cpu_count()))
        else:
            pool = mp.Pool(num_processes)

        pc = PlanCost(cost_model)
        costs, opt_costs = pc.compute_costs(qreps, preds, pool=pool)
        pool.close()
        return costs
