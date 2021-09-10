import pickle
import argparse
import glob
import pdb
import psycopg2 as pg
import time
import subprocess as sp
import os
import pandas as pd
from collections import defaultdict
import sys
sys.path.append(".")
from query_representation.utils import *
import pdb
from cost_model import *

TIMEOUT_CONSTANT = 909

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--cost_model", type=str, required=False,
            default="cm1")
    parser.add_argument("--explain", type=int, required=False,
            default=1)
    parser.add_argument("--timeout", type=int, required=False,
            default=900000)
    parser.add_argument("--rerun_timeouts", type=int, required=False,
            default=0)
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--costs_fn", type=str, required=False,
            default="PostgresPlanCost.csv")

    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="imdb")
    parser.add_argument("--pwd", type=str, required=False,
            default="password")
    parser.add_argument("--port", type=str, required=False,
            default=5432)

    return parser.parse_args()

def execute_sql(sql, cost_model="cm1",
        explain=False,
        materialize=False, timeout=900000):
    '''
    '''

    if explain:
        sql = sql.replace("explain (format json)", "explain (analyze,costs, format json)")
    else:
        sql = sql.replace("explain (format json)", "")

    con = pg.connect(port=args.port,dbname=args.db_name,
            user=args.user,password=args.pwd,host="localhost")

    # TODO: clear cache

    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute("SET geqo_threshold = {}".format(32))
    set_cost_model(cursor, cost_model)

    # the idea is that we inject the #rows using the pg_hint_plan into
    # postgresql; then the join orders etc. are computed the postgresql engine
    # using the given cardinalities
    cursor.execute("SET join_collapse_limit = {}".format(32))
    cursor.execute("SET from_collapse_limit = {}".format(32))
    cursor.execute("SET statement_timeout = {}".format(timeout))

    start = time.time()

    try:
        cursor.execute(sql)
    except Exception as e:
        cursor.execute("ROLLBACK")
        con.commit()
        if not "timeout" in str(e):
            print("failed to execute for reason other than timeout")
            print(e)
            print(sql)
            cursor.close()
            con.close()
            return None, timeout/1000 + 9.0
        else:
            print("failed because of timeout!")
            end = time.time()
            print("took {} seconds".format(end-start))

            if explain:
                sql = sql.replace("explain (analyze,costs, format json)",
                "explain (format json)")
            else:
                sql = "explain (format json) " + sql

            set_cost_model(cursor, cost_model, materialize)
            cursor.execute("SET join_collapse_limit = {}".format(1))
            cursor.execute("SET from_collapse_limit = {}".format(1))
            cursor.execute(sql)
            explain_output = cursor.fetchall()
            cursor.close()
            con.close()
            return explain_output, (timeout/1000) + 9.0

    explain_output = cursor.fetchall()
    end = time.time()

    print("took {} seconds".format(end-start))
    sys.stdout.flush()

    return explain_output, end-start

def main():
    def add_runtime_row(qname, rt, exp_analyze):
        cur_runtimes["qname"].append(qname)
        cur_runtimes["runtime"].append(rt)
        cur_runtimes["exp_analyze"].append(exp_analyze)

    cost_model = args.cost_model
    costs_fn = os.path.join(args.result_dir, args.costs_fn)

    assert os.path.exists(costs_fn)

    costs = pd.read_csv(costs_fn)
    assert isinstance(costs, pd.DataFrame)

    rt_fn = os.path.join(args.result_dir, "Runtimes.csv")

    # go in order and execute runtimes...
    if os.path.exists(rt_fn):
        runtimes = pd.read_csv(rt_fn)
    else:
        runtimes = None

    if runtimes is None:
        columns = ["qname", "runtime", "exp_analyze"]
        runtimes = pd.DataFrame(columns=columns)

    cur_runtimes = defaultdict(list)

    for i,row in costs.iterrows():
        if row["qname"] in runtimes["qname"].values:
            # what is the stored value for this key?
            rt_df = runtimes[runtimes["qname"] == row["qname"]]
            stored_rt = rt_df["runtime"].values[0]
            if stored_rt == TIMEOUT_CONSTANT and args.rerun_timeouts:
                print("going to rerun timed out query")
            else:
                print("skipping {} with stored runtime".format(row["qname"]))
                continue

        exp_analyze, rt = execute_sql(row["exec_sql"],
                cost_model=cost_model,
                explain=args.explain,
                timeout=args.timeout)

        add_runtime_row(row["qname"], rt, exp_analyze)

        rts = cur_runtimes["runtime"]
        print("#Queries:{}, AvgRt: {}".format(len(rts),
            sum(rts) / len(rts)))
        df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
        df.to_csv(rt_fn, index=False)

if __name__ == "__main__":
    args = read_flags()
    main()
