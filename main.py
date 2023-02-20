import sys
sys.path.append(".")
# from query_representation.query import *
from query_representation.utils import get_query_splits

from cardinality_estimation.featurizer import *
from cardinality_estimation import get_alg
from evaluation.eval_fns import get_eval_fn
# import glob
import argparse
# import random
import json

import pdb
import copy
import pickle
import os
import yaml

import wandb
import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

def eval_alg(alg, eval_funcs, qreps, cfg,
        samples_type,
        featurizer=None):
    '''
    '''
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    start = time.time()
    alg_name = alg.__str__()
    exp_name = alg.get_exp_name()

    ests = alg.test(qreps)

    rdir = None
    if args.result_dir is not None:
        rdir = os.path.join(args.result_dir, exp_name)
        make_dir(rdir)
        # print("Going to store results at: ", rdir)
        args_fn = os.path.join(rdir, "cfg.json")
        with open(args_fn, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

    if samples_type != "train" and cfg["eval"]["save_test_preds"]:
        preds_dir = os.path.join(rdir, samples_type + "-preds")
        make_dir(preds_dir)
        for i,qrep in enumerate(qreps):
            newfn = os.path.basename(qrep["name"])
            predfn = os.path.join(preds_dir, qrep["name"])
            cur_ests = ests[i]
            with open(predfn, "wb") as f:
                pickle.dump(cur_ests, f)

    for efunc in eval_funcs:
        if "plan" in str(efunc).lower() and "train" in qreps[0]["template_name"]:
            print("skipping _train_ workload plan cost eval")
            continue

        errors = efunc.eval(qreps, ests,
                user = cfg["db"]["user"], pwd = cfg["db"]["pwd"],
                port = cfg["db"]["port"], db_name = cfg["db"]["db_name"],
                db_host = cfg["db"]["db_host"],
                samples_type = samples_type,
                num_processes = cfg["eval"]["num_processes"],
                alg_name = alg_name,
                save_pdf_plans= cfg["eval"]["save_pdf_plans"],
                query_dir = cfg["data"]["query_dir"],
                result_dir = args.result_dir,
                use_wandb = cfg["eval"]["use_wandb"],
                featurizer = featurizer, alg=alg)

        print("{}, {}, {}, #samples: {}, {}: mean: {}, median: {}, 99p: {}"\
                .format(cfg["db"]["db_name"], samples_type, alg, len(errors),
                    efunc.__str__(), np.round(np.mean(errors),3),
                    np.round(np.median(errors),3),
                    np.round(np.percentile(errors,99),3)))

        if cfg["eval"]["use_wandb"]:
            loss_key = "Final-{}-{}-{}".format(str(efunc), samples_type,
                    "mean")
            wandb.run.summary[loss_key] = np.round(np.mean(errors),3)

    print("All loss computations took: ", time.time()-start)

def get_featurizer(trainqs, valqs, testqs, eval_qs):

    featurizer = Featurizer(**cfg["db"])
    featdata_fn = os.path.join(cfg["data"]["query_dir"],
            "dbdata.json")

    all_evalqs = []
    for e0 in eval_qs:
        all_evalqs += e0

    if args.regen_featstats or not os.path.exists(featdata_fn):
        # we can assume that we have db stats for any column in the db
        featurizer.update_column_stats(trainqs+valqs+testqs+all_evalqs)
        ATTRS_TO_SAVE = ['aliases', 'cmp_ops', 'column_stats', 'joins',
                'max_in_degree', 'max_joins', 'max_out_degree', 'max_preds',
                'max_tables', 'regex_cols', 'tables', 'join_key_stats',
                'primary_join_keys', 'join_key_normalizers',
                'join_key_stat_names', 'join_key_stat_tmps'
                'max_tables', 'regex_cols', 'tables',
                'mcvs']

        featdata = {}
        for k in dir(featurizer):
            if k not in ATTRS_TO_SAVE:
                continue
            attrvals = getattr(featurizer, k)
            if isinstance(attrvals, set):
                attrvals = list(attrvals)
            featdata[k] = attrvals

        if args.save_featstats:
            f = open(featdata_fn, "w")
            json.dumps(featdata, f)
            f.close()
    else:
        f = open(featdata_fn, "r")
        featdata = json.load(f)
        f.close()
        featurizer.update_using_saved_stats(featdata)

    if args.alg in ["mscn", "mscn_joinkey", "mstn"]:
        feat_type = "set"
    else:
        feat_type = "combined"

    card_type = "subplan"

    # Look at the various keyword arguments to setup() to change the
    # featurization behavior; e.g., include certain features etc.
    # these configuration properties do not influence the basic statistics
    # collected in the featurizer.update_column_stats call; Therefore, we don't
    # include this in the cached version

    qdir_name = os.path.basename(cfg["data"]["query_dir"])
    bitmap_dir = cfg["data"]["bitmap_dir"]
    # ** converts the dictionary into keyword args
    featurizer.setup(
            **cfg["featurizer"],
            loss_func = cfg["model"]["loss_func_name"],
            featurization_type = feat_type,
            bitmap_dir = cfg["data"]["bitmap_dir"],
            card_type = card_type
            )

    # just updates stuff like max-num-tables etc. for some implementation
    # things
    featurizer.update_max_sets(trainqs+valqs+testqs+all_evalqs)
    featurizer.update_workload_stats(trainqs+valqs+testqs+all_evalqs)

    featurizer.init_feature_mapping()

    if cfg["featurizer"]["feat_onlyseen_maxy"]:
        featurizer.update_ystats(trainqs,
                max_num_tables=cfg["model"]["max_num_tables"])
    else:
        featurizer.update_ystats(trainqs+valqs+testqs+all_evalqs,
                max_num_tables = cfg["model"]["max_num_tables"])

    featurizer.update_seen_preds(trainqs)

    return featurizer

def main():
    global args,cfg

    with open(args.config) as f:
        cfg = yaml.safe_load(f.read())

    print(yaml.dump(cfg, default_flow_style=False))

    # set up wandb logging metrics
    if cfg["eval"]["use_wandb"]:
        wandbcfg = {}
        for k,v in cfg.items():
            if isinstance(v, dict):
                for k2,v2 in v.items():
                    wandbcfg.update({k+"-"+k2:v2})
            else:
                wandbcfg.update({k:v})

        wandbcfg.update(vars(args))
        # additional config tags
        wandb_tags = ["1a"]
        if args.wandb_tags is not None:
            wandb_tags += args.wandb_tags.split(",")
        wandb.init("ceb", config=wandbcfg,
                tags=wandb_tags)

    train_qfns, test_qfns, val_qfns, eval_qfns = get_query_splits(cfg["data"])

    trainqs = load_qdata(train_qfns)
    # Note: can be quite memory intensive to load them all; might want to just
    # keep around the qfns and load them as needed
    valqs = load_qdata(val_qfns)
    testqs = load_qdata(test_qfns)

    eval_qdirs = cfg["data"]["eval_query_dir"].split(",")
    evalqs = []
    for eval_qfn in eval_qfns:
        evalqs.append(load_qdata(eval_qfn))
    eqs = [len(eq) for eq in evalqs]
    print("""Selected Queries: {} train, {} test, {} val, {} eval"""\
            .format(len(trainqs), len(testqs), len(valqs), sum(eqs)))

    # only needs featurizer for learned models
    if args.alg in ["xgb", "fcnn", "mscn", "mscn_joinkey", "mstn"]:
        featurizer = get_featurizer(trainqs, valqs, testqs, evalqs)
    else:
        featurizer = None

    alg = get_alg(args.alg, cfg)

    eval_fns = []
    for efn in args.eval_fns.split(","):
        eval_fns.append(get_eval_fn(efn))

    if cfg["model"]["eval_epoch"] < cfg["model"]["max_epochs"]:
        alg.train(trainqs, valqs=valqs, testqs=testqs, evalqs = evalqs,
                eval_qdirs = eval_qdirs, featurizer=featurizer)
    else:
        alg.train(trainqs, valqs=valqs, testqs=None, evalqs = None,
                eval_qdirs = eval_qdirs, featurizer=featurizer)

    eval_alg(alg, eval_fns, trainqs, cfg, "train", featurizer=featurizer)

    if len(valqs) > 0:
        eval_alg(alg, eval_fns, valqs, cfg, "val", featurizer=featurizer)

    if len(testqs) > 0:
        eval_alg(alg, eval_fns, testqs, cfg, "test", featurizer=featurizer)

    if len(evalqs) > 0 and len(evalqs[0]) > 0:
        for ei, evalq in enumerate(evalqs):
            eval_alg(alg, eval_fns, evalq, cfg, eval_qdirs[ei], featurizer=featurizer)
            del evalq[:]

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False,
            default="configs/config.yaml")
    parser.add_argument("--alg", type=str, required=False,
            default="mscn")

    parser.add_argument("--regen_featstats", type=int, required=False,
            default=0)
    parser.add_argument("--save_featstats", type=int, required=False,
            default=0)
    parser.add_argument("--use_saved_feats", type=int, required=False,
            default=1)

    # logging arguments
    parser.add_argument("--wandb_tags", type=str, required=False,
        default=None, help="additional tags for wandb logs")

    parser.add_argument("--result_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--eval_fns", type=str, required=False,
            default="ppc,qerr")

    return parser.parse_args()

if __name__ == "__main__":
    args = read_flags()
    main()
