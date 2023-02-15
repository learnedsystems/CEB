import pandas as pd
import os
import pickle
import numpy as np

CEB_KEY_FN = "/Users/pari/prism-testbed/ceb_runtime_qnames.pkl"

def load_object(file_name):
    res = None
    if ".csv" in file_name:
        res = pd.read_csv(file_name, sep="|", encoding='utf-8')
    else:
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                res = pickle.loads(f.read())
    return res

def get_eval_wk(trainds, st, exp_args):
    if st in ["train", "val", "test"]:
        # if trainds in ["True", "Postgres"]:
            # return exp_args["query_dir"]
        # else:
        return trainds
    elif "joblight" in st:
        return "JOB-light"
    elif "imdb" in st:
        return "CEB"
    elif "job" in st:
        return "JOB"
    else:
        return "Unknown"

def get_training_ds(exp_args):
    if exp_args["algs"] == "true":
        return "true"
    elif exp_args["algs"] == "postgres":
        return "postgres"

    if "joblight" in exp_args["query_dir"]:
        return "JOB-light"
    elif "imdb" in exp_args["query_dir"]:
        return "CEB"
    elif "job" in exp_args["query_dir"]:
        return "JOB"

def get_alg_name(exp_args):
    if exp_args["algs"] == "true":
        return "True"
    elif exp_args["algs"] == "postgres":
        return "PostgreSQL"

    assert exp_args["algs"] == "mscn"
    if exp_args["sample_bitmap"] == 1 and \
        exp_args["onehot_dropout"] == 0:
        if exp_args["loss_func_name"] == "flowloss":
            return "MSCN\n(FlowLoss)"
        else:
            return "MSCN"

    elif exp_args["join_bitmap"] == 1 and \
        exp_args["onehot_dropout"] == 2:
        if exp_args["loss_func_name"] == "flowloss":
            return "Robust-MSCN\n(FlowLoss)"
        else:
            return "Robust-MSCN"
    ## temporary
    elif exp_args["join_bitmap"] == 0 and \
         exp_args["onehot_dropout"] == 2 and \
         exp_args["train_test_split_kind"] == "custom":
        return "Robust-MSCN"
    elif exp_args["onehot_dropout"] == 0 and \
         exp_args["train_test_split_kind"] == "custom":
        return "MSCN"
    else:
        return exp_args["algs"]

def get_db_year(exp_args):
    if exp_args["algs"] == "true":
        return "Full DB"
    elif exp_args["algs"] == "postgres":
        return "Full DB"

    if "1950" in exp_args["query_dir"]:
        return "1950"

    elif "1980" in exp_args["query_dir"]:
        return "1980"
    else:
        return "Full DB"


def get_ablation_name(exp_args):
    if exp_args["algs"] == "true":
        return "True"
    elif exp_args["algs"] == "postgres":
        return "PostgreSQL"

    assert exp_args["algs"] == "mscn"
    if exp_args["sample_bitmap"] == 0 \
            and exp_args["join_bitmap"] == 0 \
            and exp_args["onehot_dropout"] == 2 \
            and exp_args["onehot_mask_truep"] == 0.8:
        return "No Sampling Features"
    elif exp_args["heuristic_features"] == 0 \
            and exp_args["join_bitmap"] == 1 \
            and exp_args["flow_features"] == 0:
        return "No Data Features"
    elif exp_args["onehot_mask_truep"] == 0.0 \
            and exp_args["join_bitmap"] == 1 \
            and exp_args["onehot_dropout"] == 2:
        return "No Query Features"
    elif exp_args["onehot_mask_truep"] == 0.0 \
            and exp_args["join_bitmap"] == 0 \
            and exp_args["sample_bitmap"] == 0 \
            and exp_args["onehot_dropout"] == 2:
        # return "No Query Driven Features,\n No Sampling Features"
        return "Only Data Features"
    elif exp_args["heuristic_features"] == 0 \
            and exp_args["join_bitmap"] == 0 \
            and exp_args["flow_features"] == 0:
        # return "No Data Driven Features,\nNo Sampling Features"
        return "Unknown"
    elif exp_args["sample_bitmap"] == 1 \
            and exp_args["join_bitmap"] == 0 \
            and exp_args["onehot_dropout"] == 2:
        return "Sample Bitmap,\nNo Join Bitmap"
    elif exp_args["sample_bitmap"] == 0 \
            and exp_args["join_bitmap"] == 1 \
            and exp_args["onehot_dropout"] == 0 \
            and exp_args["inp_dropout"] == 0: \
        return "No Query Masking"
    elif exp_args["sample_bitmap"] == 0 \
            and exp_args["join_bitmap"] == 1 \
            and exp_args["inp_dropout"] == 0.2:
        return "Mask All Features"
    elif exp_args["sample_bitmap"] == 0 \
            and exp_args["join_bitmap"] == 1 \
            and exp_args["onehot_dropout"] == 2:
        return "Robust-MSCN\n (Best Model)"
    else:
        return "Unknown"

def load_all_qerrs(results_dir, rt_kind=None):
    alldfs = []
    fns = os.listdir(results_dir)
    qfn = "QError.csv"

    for fn in fns:
        cur_dir = results_dir + "/" + fn + "/"
        if not os.path.isdir(cur_dir):
            continue

        exp_args = load_object(cur_dir + "/args.pkl")
        if exp_args is None:
            print("missing args")
            continue
        else:
            exp_args = vars(exp_args)

        if os.path.exists(cur_dir + qfn):
            qerrs = pd.read_csv(cur_dir+qfn)
        else:
            print("missing qerrs")
            continue

        if rt_kind is not None and "qname" in qerrs.keys():
            if rt_kind.lower() == "ceb":
                qnames = load_object(CEB_KEY_FN)
                qerrs = qerrs[qerrs["qname"].isin(qnames)]
            else:
                qnames = load_object(CEB_KEY_FN)
                qerrs = qerrs[~qerrs["qname"].isin(qnames)]

        if "qname" in qerrs.keys():
            qerrs = qerrs[["qname", "samples_type", "qidx", "errors"]]
        else:
            qerrs = qerrs[["samples_type", "errors"]]

        alg = get_alg_name(exp_args)
        ablation_kind = get_ablation_name(exp_args)
        db_year = get_db_year(exp_args)

        if alg in ["True", "PostgreSQL"]:
            qerrs["samples_type"] = exp_args["query_dir"]
            print(alg)
        else:
            qerrs = qerrs[qerrs["samples_type"] != "train"]

        qerrs["alg_dir"] = fn

        trainds = get_training_ds(exp_args)

        qerrs["alg"] = alg
        qerrs["trainds"] = trainds
        qerrs["ablation_kind"] = ablation_kind
        qerrs["db_year"] = db_year

        ## need to divide into train / eval wks
        qdfs = []
        for st in set(qerrs["samples_type"]):
            tmp = qerrs[qerrs["samples_type"] == st]
            tmp["eval_workload"] = get_eval_wk(trainds, st, exp_args)
            qdfs.append(tmp)
        qerrs = pd.concat(qdfs)

        ARG_KEYS = ["sample_bitmap", "hidden_layer_size",
                "flow_features",
                "join_bitmap",
                "max_discrete_featurizing_buckets"]

        for k in ARG_KEYS:
            qerrs[k] = exp_args[k]

        alldfs.append(qerrs)

    return pd.concat(alldfs)

def load_all_runtimes(results_dir, rt_kind=None):
    alldfs = []
    fns = os.listdir(results_dir)
    rt_fn = "Runtimes.csv"

    for fn in fns:
        cur_dir = results_dir + "/" + fn + "/"
        if not os.path.isdir(cur_dir):
            continue

        exp_args = load_object(cur_dir + "/args.pkl")
        if exp_args is None:
            print("missing args")
            continue
        else:
            exp_args = vars(exp_args)

        if os.path.exists(cur_dir + rt_fn):
            runtimes = pd.read_csv(cur_dir+rt_fn)
        else:
            continue

        if rt_kind is not None:
            if rt_kind.lower() == "ceb":
                qnames = load_object(CEB_KEY_FN)
                runtimes = runtimes[runtimes["qname"].isin(qnames)]
            else:
                qnames = load_object(CEB_KEY_FN)
                runtimes = runtimes[~runtimes["qname"].isin(qnames)]

        runtimes = runtimes[["qname", "runtime"]]
        runtimes["alg_dir"] = fn
        alg = get_alg_name(exp_args)
        ablation_kind = get_ablation_name(exp_args)
        db_year = get_db_year(exp_args)

        trainds = get_training_ds(exp_args)

        runtimes["alg"] = alg
        runtimes["trainds"] = trainds
        runtimes["ablation_kind"] = ablation_kind
        runtimes["db_year"] = db_year

        if exp_args["query_dir"] == "queries/joblight_train" \
                and ("1950" not in db_year and "1980" not in db_year) \
                and len(runtimes) == 113:
            runtimes["runtime"] += 1.0

        ARG_KEYS = ["sample_bitmap", "hidden_layer_size",
                "flow_features",
                "join_bitmap",
                "max_discrete_featurizing_buckets",
                "query_dir",
                "random_bitmap_idx"
                ]

        for k in ARG_KEYS:
            if k not in exp_args:
                runtimes[k] = None
            else:
                runtimes[k] = exp_args[k]

        alldfs.append(runtimes)

    return pd.concat(alldfs)
