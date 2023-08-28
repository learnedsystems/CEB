import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plan_losses import PPC, PlanCost,get_leading_hint
from .cost_model import *
# from query_representation.utils import deterministic_hash,make_dir

from query_representation.viz import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import multiprocessing as mp
import random
from collections import defaultdict
import pandas as pd
import networkx as nx
import os
import wandb
import pickle

import pdb

TIMEOUT_CARD = 15000100000

def get_eval_fn(loss_name):
    if loss_name == "qerr":
        return QError()
    elif loss_name == "qerr_joinkey":
        return QErrorJoinKey()
    elif loss_name == "abs":
        return AbsError()
    elif loss_name == "rel":
        return RelativeError()
    elif loss_name == "ppc":
        return PostgresPlanCost(cost_model="C")
    elif loss_name == "ppc2":
        return PostgresPlanCost(cost_model="C2")
    elif loss_name == "plancost":
        return SimplePlanCost()
    elif loss_name == "flowloss":
        return FlowLoss()
    elif loss_name == "constraints":
        return LogicalConstraints()
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
        # assert len(qrep) == len(pred_subsets)

        keys = list(pred_subsets.keys())
        if SOURCE_NODE in keys:
            keys.remove(SOURCE_NODE)
        assert len(keys) == len(pred_subsets)

        keys.sort()
        for alias in keys:
            pred = pred_subsets[alias]
            if "actual" not in qrep[alias]["cardinality"]:
                assert False
                # continue

            actual = qrep[alias]["cardinality"]["actual"]
            if actual < 0:
                actual = 1
                pred = 1
                # continue

            # if actual >= TIMEOUT_CARD:
                # actual = TIMEOUT_CARD
                # pred = TIMEOUT_CARD

            if actual == 0:
                actual += 1
            if pred <= 0:
                pred = 1

            ytrue.append(float(actual))
            yhat.append(float(pred))

    return np.array(ytrue), np.array(yhat)

def _get_all_joinkeys(qreps, preds):
    ytrue = []
    yhat = []

    for i, joinkeys in enumerate(preds):
        einfos = qreps[i]["subset_graph"].edges()
        keys = list(joinkeys.keys())
        keys.sort()

        for curkey in keys:
            pred = joinkeys[curkey]
            # actual = einfos[curkey]["actual"]
            jcards = einfos[curkey]["join_key_cardinality"]
            actual = list(jcards.values())[0]["actual"]

            if actual == 0:
                actual += 1
            if pred == 0:
                pred += 1

            ytrue.append(float(actual))
            yhat.append(float(pred))

    return np.array(ytrue), np.array(yhat)

class LogicalConstraints(EvalFunc):
    def __init__(self, **kwargs):
        pass

    def save_logs(self, qreps, errors, **kwargs):
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
        errors = []
        id_errs = []
        fkey_errs = []

        featurizer = kwargs["featurizer"]

        for qi, qrep in enumerate(qreps):
            cur_errs = []

            cur_preds = preds[qi]
            sg = qrep["subset_graph"]
            jg = qrep["join_graph"]
            for node in sg.nodes():
                if node == SOURCE_NODE:
                    continue

                edges = sg.out_edges(node)
                nodepred = cur_preds[node]
                # calculating error per node instead of per edge
                error = 0

                for edge in edges:
                    prev_node = edge[1]
                    newt = list(set(edge[0]) - set(edge[1]))[0]
                    tab_pred = cur_preds[(newt,)]
                    for alias in edge[1]:
                        if (alias,newt) in jg.edges():
                            jdata = jg.edges[(alias,newt)]
                        elif (newt,alias) in jg.edges():
                            jdata = jg.edges[(newt,alias)]
                        else:
                            continue
                        if newt not in jdata or alias not in jdata:
                            continue

                        newjkey = jdata[newt]
                        otherjkey = jdata[alias]

                        if not featurizer.feat_separate_alias:
                            newjkey = ''.join([ck for ck in newjkey if not ck.isdigit()])
                            otherjkey = ''.join([ck for ck in otherjkey if not ck.isdigit()])

                        stats1 = featurizer.join_key_stats[newjkey]
                        stats2 = featurizer.join_key_stats[otherjkey]

                        newjcol = newjkey[newjkey.find(".")+1:]
                        if newjcol == "id":
                            card1 = cur_preds[(newt,)]
                            maxfkey = stats2["max_key"]
                            maxcard1 = maxfkey*card1

                            ## FIXME: not fully accurate
                            if cur_preds[node] > maxcard1:
                                fkey_errs.append(1.0)
                            else:
                                fkey_errs.append(0.0)

                            # could not have got bigger
                            if cur_preds[prev_node] < cur_preds[node]:
                                error = 1
                                id_errs.append(1)
                            else:
                                id_errs.append(0)

                        # else:
                            # # new table was a foreign key
                            # maxfkey = stats1["max_key"]
                            # card_prev = cur_preds[prev_node]
                            # maxcurcard = card_prev * maxfkey
                            # if maxcurcard < cur_preds[node]:
                                # print("BAD")
                                # pdb.set_trace()

                cur_errs.append(error)
            errors.append(np.mean(cur_errs))

        print("pkey x fkey errors: ", np.mean(fkey_errs), np.sum(fkey_errs))
        print("primary key id errors: ", np.mean(id_errs))
        return errors

    def __str__(self):
        return self.__class__.__name__

class QErrorJoinKey(EvalFunc):

    def eval(self, qreps, preds, **kwargs):
        '''
        '''
        assert len(preds) == len(qreps)
        assert isinstance(preds[0], dict)

        ytrue, yhat = _get_all_joinkeys(qreps, preds)

        assert len(ytrue) == len(yhat)

        errors = np.maximum((ytrue / yhat), (yhat / ytrue))

        num_table_errs = defaultdict(list)
        didx = 0

        for i, qrep in enumerate(qreps):
            edges = list(qrep["subset_graph"].edges())
            # if SOURCE_NODE in nodes:
                # nodes.remove(SOURCE_NODE)
            edges.sort(key = lambda x: str(x))
            for qi, edge in enumerate(edges):
                assert len(edge[1]) < len(edge[0])
                numt = len(edge[1])
                curerr = errors[didx]
                num_table_errs[numt].append(curerr)
                didx += 1

        nts = list(num_table_errs.keys())

        nts.sort()
        for nt in nts:
            print("{} Tables, JoinKey-QError mean: {}, 99p: {}".format(
                nt, np.mean(num_table_errs[nt]),
                np.percentile(num_table_errs[nt], 99)))

        return errors

class QError(EvalFunc):

    def save_logs(self, qreps, errors, **kwargs):
        result_dir = kwargs["result_dir"]
        if result_dir is None:
            return

        if "samples_type" in kwargs:
            samples_type = kwargs["samples_type"]
        else:
            samples_type = ""

        num_table_errs = defaultdict(list)
        didx = 0
        qnames = []
        qidxs = []

        for i, qrep in enumerate(qreps):
            nodes = list(qrep["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            nodes.sort()
            for qi, node in enumerate(nodes):
                numt = len(node)
                if didx >= len(errors):
                    # assert False
                    continue
                qnames.append(qrep["name"])
                qidxs.append(qi)
                curerr = errors[didx]
                cards = qrep["subset_graph"].nodes()[node]["cardinality"]

                num_table_errs[numt].append(curerr)
                didx += 1

        resfn = os.path.join(result_dir, self.__str__() + ".csv")

        res = pd.DataFrame(data=errors, columns=["errors"])
        res["samples_type"] = samples_type
        res["qname"] = qnames
        res["qidx"] = qidxs

        # TODO: add other data?
        if os.path.exists(resfn):
            res.to_csv(resfn, mode="a",header=False)
        else:
            res.to_csv(resfn, header=True)

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

        if kwargs["result_dir"] is not None:
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

class MeanSquaredError(EvalFunc):
    def eval(self, qreps, preds, **kwargs):
        '''
        '''
        assert len(preds) == len(qreps)
        assert isinstance(preds[0], dict)

        ytrue, yhat = _get_all_cardinalities(qreps, preds)
        errors = np.linalg.mse(yhat - ytrue)
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
    def __init__(self, cost_model="C"):
        self.cost_model = cost_model

    def __str__(self):
        return self.__class__.__name__ + "-" + self.cost_model

    def save_logs(self, qreps, errors, **kwargs):
        if "result_dir" not in kwargs:
            return

        result_dir = kwargs["result_dir"]

        if result_dir is None:
            return

        if "save_pdf_plans" in kwargs:
            save_pdf_plans = kwargs["save_pdf_plans"]
        else:
            save_pdf_plans = False

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

        if result_dir is not None:
            costs_fn = os.path.join(result_dir, self.__str__() + ".csv")

            if os.path.exists(costs_fn):
                costs_df = pd.read_csv(costs_fn)
            else:
                columns = ["qname", "join_order", "exec_sql", "cost",
                        "samples_type"]
                costs_df = pd.DataFrame(columns=columns)

            cur_costs = defaultdict(list)

            for i, qrep in enumerate(qreps):
                qname = os.path.basename(qrep["name"])
                cur_costs["qname"].append(qname)

                joinorder = get_leading_hint(qrep["join_graph"], plans[i])
                cur_costs["join_order"].append(joinorder)

                cur_costs["exec_sql"].append(sqls[i])
                cur_costs["cost"].append(costs[i])
                cur_costs["samples_type"].append(samples_type)

            cur_df = pd.DataFrame(cur_costs)
            combined_df = pd.concat([costs_df, cur_df], ignore_index=True)
            combined_df.to_csv(costs_fn, index=False)

        # FIXME: hard to append to pdfs, so use samples_type to separate
        # out the different times this function is currently called.

        if save_pdf_plans:
            pdffn = samples_type + "_query_plans.pdf"
            pdf = PdfPages(os.path.join(result_dir, pdffn))
            for i, plan in enumerate(plans):
                if plan is None:
                    continue
                # we know cost of this; we know best cost;
                title_fmt = """{}. PostgreSQL Plan Cost w/ True Cardinalities: {}\n; PostgreSQL Plan Cost w/ {} Estimates: {}\n PostgreSQL Plan using {} Estimates"""

                title = title_fmt.format(qreps[i]["name"], opt_costs[i],
                        alg_name, costs[i], alg_name)

                plot_explain_join_order(plan[0][0][0], true_cardinalities[i],
                        est_cardinalities[i], pdf, title)

            pdf.close()

        # Total costs
        totalcost = np.sum(costs)
        opttotal = np.sum(opt_costs)

        relcost = np.round(float(totalcost)/opttotal, 3)

        ppes = costs - opt_costs

        print("{}, {}. All templates. #samples: {}, Relative Postgres Cost: {}"\
                .format(samples_type, alg_name, len(costs),
                    relcost))

        template_costs = defaultdict(list)
        true_template_costs = defaultdict(list)
        tmp_rel_costs = {}
        tmp_avg_errs = {}

        for ci in range(len(costs)):
            template = qreps[ci]["template_name"]
            template_costs[template].append(costs[ci])
            true_template_costs[template].append(opt_costs[ci])

        for tmp in template_costs:
            tmp_costs = np.array(template_costs[tmp])
            tmp_opt_costs = np.array(true_template_costs[tmp])
            tmp_relc = np.round(np.sum(tmp_costs) / float(np.sum(tmp_opt_costs)), 3)
            tmp_avg_err = np.round(np.mean(tmp_costs - tmp_opt_costs), 3)
            tmp_rel_costs[tmp] = tmp_relc
            tmp_avg_errs[tmp] = tmp_avg_err

            print("Template: {}, #samples: {} Relative Postgres Cost: {}, Avg Err: {}"\
                    .format( tmp, len(tmp_costs), tmp_relc, tmp_avg_err))

    def eval(self, qreps, preds,
            user="ceb",
            pwd="password",
            port=5432,
            db_name="imdb",
            db_host="localhost",
            num_processes=-1,
            save_pdf_plans=False,
            result_dir=None, **kwargs):
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
        cost_model = self.cost_model

        if num_processes == -1:
            pool = mp.Pool(int(mp.cpu_count()))
        elif num_processes == -2:
            pool = None
        else:
            pool = mp.Pool(num_processes)
        # db_args = kwargs["db_args"]

        ppc = PPC(cost_model, user, pwd,
                db_host, port, db_name)

        est_cardinalities = []
        true_cardinalities = []
        sqls = []
        join_graphs = []

        pg_query_costs = {}
        pg_costs = []

        for i, qrep in enumerate(qreps):
            sqls.append(qrep["sql"])
            join_graphs.append(qrep["join_graph"])
            ests = {}
            trues = {}
            predq = preds[i]
            for node, node_info in qrep["subset_graph"].nodes().items():
                if node == SOURCE_NODE:
                    continue
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

        self.save_logs(qreps, costs, **kwargs,
                sqls=sqls,
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
        if "samples_type" in kwargs:
            samples_type = kwargs["samples_type"]
        else:
            samples_type = ""

        if num_processes == -1:
            pool = mp.Pool(int(mp.cpu_count()))
        else:
            pool = mp.Pool(num_processes)

        pc = PlanCost(cost_model)
        costs, opt_costs = pc.compute_costs(qreps, preds, pool=pool)
        pool.close()

        totalcost = np.sum(costs)
        opttotal = np.sum(opt_costs)
        relcost = np.round(float(totalcost)/opttotal, 3)

        print("{}, #samples: {}, Relative Plan Cost: {}"\
                .format(samples_type, len(costs),
                    relcost))

        return costs
