import sys
sys.path.append(".")
from query_representation.query import *
from evaluation.eval_fns import *
from cardinality_estimation.featurizer import *
from cardinality_estimation.algs import *

import glob
import argparse
import random
import klepto
from sklearn.model_selection import train_test_split
import pdb

def eval_alg(alg, eval_funcs, qreps, samples_type):
    '''
    '''
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    start = time.time()
    alg_name = alg.__str__()
    exp_name = alg.get_exp_name()
    ests = alg.test(qreps)

    for efunc in eval_funcs:
        rdir = None
        if args.result_dir is not None:
            rdir = os.path.join(args.result_dir, exp_name)
            make_dir(rdir)

        errors = efunc.eval(qreps, ests, args=args, samples_type=samples_type,
                result_dir=rdir, user = args.user, db_name = args.db_name,
                db_host = args.db_host, port = args.port,
                alg_name = alg_name)

        if args.result_dir is not None:
            resfn = os.path.join(rdir, efunc.__str__() + ".csv")
            res = pd.DataFrame(data=errors, columns=["errors"])
            res["samples_type"] = samples_type
            # todo: add other data?
            if os.path.exists(resfn):
                res.to_csv(resfn, mode="a",header=False)
            else:
                res.to_csv(resfn, header=True)

        print("{}, {}, {}, #samples: {}, {}: mean: {}, median: {}, 99p: {}"\
                .format(args.db_name, samples_type, alg, len(errors),
                    efunc.__str__(),
                    np.round(np.mean(errors),3),
                    np.round(np.median(errors),3),
                    np.round(np.percentile(errors,99),3)))

    print("all loss computations took: ", time.time()-start)

def get_alg(alg):

    if alg == "saved":
        assert args.model_dir is not None
        return SavedPreds(model_dir=args.model_dir)
    elif alg == "postgres":
        return Postgres()
    elif alg == "true":
        return TrueCardinalities()
    elif alg == "true_rank":
        return TrueRank()
    elif alg == "true_random":
        return TrueRandom()
    elif alg == "true_rank_tables":
        return TrueRankTables()
    elif alg == "random":
        return Random()
    elif alg == "rf":
        return RandomForest(grid_search = False,
                n_estimators = 100,
                max_depth = 10,
                lr = 0.01)
    elif alg == "xgb":
        return XGBoost(grid_search=False, tree_method="hist",
                       subsample=1.0, n_estimators = 100,
                       max_depth=10, lr = 0.01)
    else:
        assert False

def get_query_fns():
    fns = list(glob.glob(args.query_dir + "/*"))
    skipped_templates = []
    train_qfns = []
    test_qfns = []
    val_qfns = []

    for qi,qdir in enumerate(fns):
        template_name = os.path.basename(qdir)
        if args.query_templates != "all":
            query_templates = args.query_templates.split(",")
            if template_name not in query_templates:
                skipped_templates.append(template_name)
                continue

        # let's first select all the qfns we are going to load
        qfns = list(glob.glob(qdir+"/*.pkl"))
        qfns.sort()

        if args.num_samples_per_template == -1:
            qfns = qfns
        elif args.num_samples_per_template < len(qfns):
            qfns = qfns[0:args.num_samples_per_template]
        else:
            assert False

        if args.test_diff_templates:
            cur_val_fns = []
            assert False
        else:
            cur_val_fns, qfns = train_test_split(qfns,
                    test_size=1-args.val_size,
                    random_state=args.seed)
            cur_train_fns, cur_test_fns = train_test_split(qfns,
                    test_size=args.test_size,
                    random_state=args.seed)

        train_qfns += cur_train_fns
        val_qfns += cur_val_fns
        test_qfns += cur_test_fns

    print("skipped templates: ", " ".join(skipped_templates))
    return train_qfns, test_qfns, val_qfns

def load_qdata(fns):
    qreps = []
    for qfn in fns:
        qrep = load_qrep(qfn)
        # TODO: can do checks like no queries with zero cardinalities etc.
        qreps.append(qrep)
        template_name = os.path.basename(os.path.dirname(qfn))
        qrep["name"] = os.path.basename(qfn)
        qrep["template_name"] = template_name

    return qreps

def get_featurizer(trainqs, valqs, testqs):
    featkey = deterministic_hash("db-" + args.query_dir + \
                args.query_templates + args.algs)
    misc_cache = klepto.archives.dir_archive("./misc_cache",
            cached=True, serialized=True)
    found_feats = featkey in misc_cache.archive and not args.regen_featstats

    # collecting statistics about each column (min/max/unique vals etc.) can
    # take a few minutes on the IMDb workload; so we cache the results
    if found_feats:
        featurizer = misc_cache.archive[featkey]
    else:
        featurizer = Featurizer(args.user, args.pwd, args.db_name,
                args.db_host, args.port)
        featurizer.update_column_stats(trainqs+valqs+testqs)
        misc_cache.archive[featkey] = featurizer

    if args.algs == "mscn":
        feat_type = "set"
    else:
        feat_type = "combined"

    # Look at the various keyword arguments to setup() to change the
    # featurization behavior; e.g., include certain features etc.
    # these configuration properties do not influence the basic statistics
    # collected in the featurizer.update_column_stats call; Therefore, we don't
    # include this in the cached version
    featurizer.setup(ynormalization=args.ynormalization)
    featurizer.update_ystats(trainqs+valqs+testqs)

    return featurizer

def main():
    train_qfns, test_qfns, val_qfns = get_query_fns()
    print("""Selected {} train queries, {} test queries, and {} val queries"""\
            .format(len(train_qfns), len(test_qfns), len(val_qfns)))
    trainqs = load_qdata(train_qfns)

    # Note: can be quite memory intensive to load them all; might want to just
    # keep around the qfns and load them as needed
    valqs = load_qdata(val_qfns)
    testqs = load_qdata(test_qfns)

    # only needs featurizer for learned models
    if args.algs in ["xgb", "fcnn", "mscn"]:
        featurizer = get_featurizer(trainqs, valqs, testqs)
    else:
        featurizer = None

    algs = []
    for alg_name in args.algs.split(","):
        algs.append(get_alg(alg_name))

    eval_fns = []
    for efn in args.eval_fns.split(","):
        eval_fns.append(get_eval_fn(efn))

    for alg in algs:
        alg.train(trainqs, valqs=valqs, testqs=testqs,
                featurizer=featurizer, result_dir=args.result_dir)
        eval_alg(alg, eval_fns, trainqs, "train")

        if len(valqs) > 0:
            eval_alg(alg, eval_fns, valqs, "val")

        if len(testqs) > 0:
            eval_alg(alg, eval_fns, testqs, "test")

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", type=str, required=False,
            default="./queries/imdb/")

    ## db credentials
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="arthurfleck")
    parser.add_argument("--pwd", type=str, required=False,
            default="password")
    parser.add_argument("--port", type=int, required=False,
            default=5432)

    parser.add_argument("--result_dir", type=str, required=False,
            default=None)
    parser.add_argument("--query_templates", type=str, required=False,
            default="all")

    parser.add_argument("--seed", type=int, required=False,
            default=13)
    parser.add_argument("--test_diff_templates", type=int, required=False,
            default=0)
    parser.add_argument("-n", "--num_samples_per_template", type=int, required=False,
            default=-1)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--val_size", type=float, required=False,
            default=0.2)
    parser.add_argument("--algs", type=str, required=False,
            default="postgres")
    parser.add_argument("--eval_fns", type=str, required=False,
            default="qerr")

    # featurizer arguments
    parser.add_argument("--regen_featstats", type=int, required=False,
            default=1)
    parser.add_argument("--ynormalization", type=str, required=False,
            default="log")

    ## NN training features
    parser.add_argument("--weight_decay", type=float, required=False,
            default=0.1)

    return parser.parse_args()

if __name__ == "__main__":
    args = read_flags()
    main()
