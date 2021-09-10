import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plan_losses import PPC, PlanCost,get_leading_hint
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
    elif loss_name == "abs":
        return AbsError()
    elif loss_name == "rel":
        return RelativeError()
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

    def save_logs(self, qreps, errors, **kwargs):
        result_dir = kwargs["result_dir"]
        if result_dir is None:
            return

        if "samples_type" in kwargs:
            samples_type = kwargs["samples_type"]
        else:
            samples_type = ""

        resfn = os.path.join(result_dir, self.__str__() + ".csv")
        res = pd.DataFrame(data=errors, columns=["errors"])
        res["samples_type"] = samples_type
        # TODO: add other data?
        if os.path.exists(resfn):
            res.to_csv(resfn, mode="a",header=False)
        else:
            res.to_csv(resfn, header=True)

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

        self.save_logs(qreps, errors, **kwargs)

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
        ytrue, yhat = _get_all_cardinalities(qreps, preds)
        # TODO: may want to choose a minimum estimate
        # epsilons = np.array([1]*len(yhat))
        # ytrue = np.maximum(ytrue, epsilons)

        errors = np.abs(ytrue - yhat) / ytrue
        return errors

class PostgresPlanCost(EvalFunc):
    def save_logs(self, qreps, errors, **kwargs):
        if "result_dir" not in kwargs:
            return

        result_dir = kwargs["result_dir"]
        if result_dir is None:
            return

        sqls = kwargs["sqls"]
        plans = kwargs["plans"]
        opt_costs = kwargs["opt_costs"]
        true_cardinalities = kwargs["true_cardinalities"]
        est_cardinalities = kwargs["est_cardinalities"]

        costs = errors

        if "samples_type" in kwargs:
            samples_type = kwargs["samples_type"]
        else:
            samples_type = ""
        if "alg_name" in kwargs:
            alg_name = kwargs["alg_name"]
        else:
            alg_name = "Est"

        costs_fn = os.path.join(result_dir, self.__str__() + ".csv")

        if os.path.exists(costs_fn):
            costs_df = pd.read_csv(costs_fn)
        else:
            columns = ["qname", "join_order", "exec_sql", "cost"]
            costs_df = pd.DataFrame(columns=columns)

        cur_costs = defaultdict(list)

        for i, qrep in enumerate(qreps):
            # sql_key = str(deterministic_hash(qrep["sql"]))
            # cur_costs["sql_key"].append(sql_key)
            qname = os.path.basename(qrep["name"])
            cur_costs["qname"].append(qname)

            joinorder = get_leading_hint(qrep["join_graph"], plans[i])
            cur_costs["join_order"].append(joinorder)

            cur_costs["exec_sql"].append(sqls[i])
            cur_costs["cost"].append(costs[i])

        cur_df = pd.DataFrame(cur_costs)
        combined_df = pd.concat([costs_df, cur_df], ignore_index=True)
        combined_df.to_csv(costs_fn, index=False)

        # FIXME: hard to append to pdfs, so use samples_type to separate
        # out the different times this function is currently called.

        pdffn = samples_type + "_query_plans.pdf"
        pdf = PdfPages(os.path.join(result_dir, pdffn))
        for i, plan in enumerate(plans):
            if plan is None:
                continue
            # we know cost of this; we know best cost;
            title_fmt = """{}. PostgreSQL Plan Cost w/ True Cardinalities: {}\n; PostgreSQL Plan Cost w/ {} Estimates: {}\n PostgreSQL Plan using {} Estimates"""

            title = title_fmt.format(qreps[i]["name"], opt_costs[i],
                    alg_name, costs[i], alg_name)

            # no idea why explains we get from cursor.fetchall() have so
            # many nested lists[][]
            plot_explain_join_order(plan[0][0][0], true_cardinalities[i],
                    est_cardinalities[i], pdf, title)

        pdf.close()

    def eval(self, qreps, preds, user="imdb",pwd="password",
            db_name="imdb", db_host="localhost", port=5432, num_processes=-1,
            result_dir=None, cost_model="cm1", **kwargs):
        ''''
        @kwargs:
            cost_model: this is just a convenient key to specify the PostgreSQL
            configuration to use. You can implement new versions in the function
            set_cost_model. e.g., cm1: disable materialization and parallelism, and
            enable all other flags.
        @ret:
            pg_costs
            Further, the following are saved in the result logs
                pg_costs:
                pg_plans: explains used to get the pg costs
                pg_sqls: sqls to execute
        '''
        assert isinstance(qreps, list)
        assert isinstance(preds, list)
        assert isinstance(qreps[0], dict)

        if num_processes == -1:
            pool = mp.Pool(int(mp.cpu_count()))
        elif num_processes == -2:
            pool = None
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

        self.save_logs(qreps, costs, **kwargs, sqls=sqls,
                plans=plans, opt_costs=opt_costs,
                true_cardinalities=true_cardinalities,
                est_cardinalities=est_cardinalities,
                result_dir=result_dir)

        if pool is not None:
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
