import argparse
import psycopg2 as pg
import sys
sys.path.append(".")
import pdb
import random
import klepto
from multiprocessing import Pool
import multiprocessing
import toml
# from db_utils.query_storage import *
from utils.utils import *
import json
import pickle

# from sql_rep.query import parse_sql
# from sql_rep.utils import *
from query_representation.query import *
from query_gen.query_generator import *
import time
import glob

def verify_queries(query_strs):
    all_queries = []
    for cur_sql in query_strs:
        start = time.time()
        test_sql = "EXPLAIN " + cur_sql
        output = cached_execute_query(test_sql, args.user,
                args.db_host, args.port, args.pwd, args.db_name,
                100, "./qgen_cache", None)
        if len(output) == 0:
            print("zero query: ", test_sql)
            continue
        else:
            print("query len: {}, time: {}".format(len(output),
                time.time()-start))
        all_queries.append(cur_sql)
    return all_queries

def remove_doubles(query_strs):
    newq = []
    seen_samples = set()
    for q in query_strs:
        if q in seen_samples:
            print(q)
            continue
        seen_samples.add(q)
        newq.append(q)
    return newq

def gen_queries(query_template, num_samples, args):
    '''
    @query_template: dict, or str, as used by QueryGenerator2 or
    QueryGenerator.
    '''
    qg = QueryGenerator(query_template, args.user, args.db_host, args.port,
		args.pwd, args.db_name)

    gen_sqls = qg.gen_queries(num_samples)
    gen_sqls = remove_doubles(gen_sqls)
    # TODO: remove queries that evaluate to zero
    return gen_sqls

def main():
    fns = list(glob.glob(args.template_dir+"/*"))
    for fn in fns:
        start = time.time()
        assert ".toml" in fn
        template_name = os.path.basename(fn).replace(".toml", "")
        # tmp_dir = qdir + template_name
        # make_dir(tmp_dir)
        out_dir = args.query_output_dir + "/" + template_name
        make_dir(out_dir)

        template = toml.load(fn)
        query_strs = gen_queries(template, args.num_samples_per_template, args)

        query_strs = verify_queries(query_strs)
        query_strs = remove_doubles(query_strs)
        print("after verifying, and removing doubles: ", len(query_strs))

        for i, sql in enumerate(query_strs):
            qrep = parse_sql(sql)
            qrep_fn = out_dir + "/" + str(deterministic_hash(sql)) + ".pkl"
            with open(qrep_fn, "wb") as f:
                pickle.dump(qrep, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_flags():
    # FIXME: simplify this stuff
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="arthurfleck")
    parser.add_argument("--pwd", type=str, required=False,
            default="password")
    parser.add_argument("--template_dir", type=str, required=False,
            default=None)
    parser.add_argument("--port", type=str, required=False,
            default=5401)
    parser.add_argument("--query_output_dir", type=str, required=False,
            default=None)
    parser.add_argument("-n", "--num_samples_per_template", type=int,
            required=False, default=10)

    parser.add_argument("--only_nonzero_samples", type=int, required=False,
            default=1)
    parser.add_argument("--random_seed", type=int, required=False,
            default=2112)

    return parser.parse_args()

args = read_flags()
main()
