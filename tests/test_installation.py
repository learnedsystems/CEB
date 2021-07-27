import sys
sys.path.append(".")
from query_representation.query import *
from losses.losses import *
import glob
import random

query_dir = "./queries/imdb/"
test_queries = ["4a/4a100.pkl"]
num_per_template=10

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

    compute_postgres_plan_cost(qreps, preds)

def test_plan_cost():
    qreps = []
    preds = []
    for q in test_queries:
        qfn = query_dir + q
        qrep = load_qrep(qfn)
        qreps.append(qrep)
        ests = get_postgres_cardinalities(qrep)
        preds.append(ests)

    compute_plan_cost(qreps, preds, cost_model="C")

test_load()
test_pg_cost()
test_plan_cost()
