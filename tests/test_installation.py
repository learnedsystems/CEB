import sys
sys.path.append(".")
from query_representation.query import *
from evaluation.eval_fns import *
import glob
import random

query_dir = "./queries/imdb/"
test_queries = ["4a/4a100.pkl", "1a/1a10.pkl"]

def test_load():
    for q in test_queries:
        qfn = query_dir + q
        qrep = load_qrep(qfn)
        tables, aliases = get_tables(qrep)
        joins = get_joins(qrep)
        preds = get_predicates(qrep)

def test_pg_cost():
    qreps = []
    preds = []
    for q in test_queries:
        qfn = query_dir + q
        qrep = load_qrep(qfn)
        qreps.append(qrep)
        ests = get_postgres_cardinalities(qrep)
        preds.append(ests)

    ppc = get_eval_fn("ppc")
    errors = ppc.eval(qreps, preds, samples_type="test",
            result_dir=None, user = "ceb", db_name = "imdb",
            db_host = "localhost", port = 5432,
            alg_name = "test")

    print("Postgres Plan Cost: ", np.mean(errors))

def test_plan_cost():
    qreps = []
    preds = []
    for q in test_queries:
        qfn = query_dir + q
        qrep = load_qrep(qfn)
        qreps.append(qrep)
        ests = get_postgres_cardinalities(qrep)
        preds.append(ests)

    pc = get_eval_fn("plancost")
    errors = pc.eval(qreps, preds, samples_type="test",
            result_dir=None, alg_name = None)
    print("Plan Cost: ", np.mean(errors))

test_load()
test_pg_cost()
test_plan_cost()
