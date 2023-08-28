import pickle
import numpy as np
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
# from query_representation.utils import *
import pdb
from cost_model import *

TIMEOUT_CONSTANT = 909

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_type", type=str, required=False,
            default=None)
    parser.add_argument("--drop_cache", type=int, required=False,
            default=0)
    parser.add_argument("--result_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--cost_model", type=str, required=False,
            default="C")
    parser.add_argument("--materialize", type=int, required=False,
            default=0)
    parser.add_argument("--explain", type=int, required=False,
            default=1)
    parser.add_argument("--reps", type=int, required=False,
            default=1)
    parser.add_argument("--timeout", type=int, required=False,
            default=None)
    parser.add_argument("--rerun_timeouts", type=int, required=False,
            default=1)
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--costs_fn_tmp", type=str, required=False,
            default="PostgresPlanCost-{}.csv")

    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="ceb")
    parser.add_argument("--pwd", type=str, required=False,
            default="password")
    parser.add_argument("--port", type=str, required=False,
            default=5432)

    return parser.parse_args()

def execute_sql(sql, cost_model="cm1",
        explain=False,
        materialize=False,
        timeout=900000,
        drop_cache=False
        ):
    '''
    '''
    if explain:
        sql = sql.replace("explain (format json)", "explain (analyze,costs, format json)")
    else:
        sql = sql.replace("explain (format json)", "")

    if drop_cache:
        drop_cache_cmd = "bash drop_cache_docker.sh"
        p = sp.Popen(drop_cache_cmd, shell=True)
        p.wait()
        time.sleep(0.1)
        for ri in range(30):
            try:
                con = pg.connect(port=args.port,dbname=args.db_name,
                        user=args.user,password=args.pwd,host="localhost")
                print("succeeded in try: ", ri)
                break
            except:
                print("failed in try: ", ri)
                time.sleep(0.1)
                continue

    else:
        con = pg.connect(port=args.port,dbname=args.db_name,
                user=args.user,password=args.pwd,host="localhost")

    # TODO: clear cache

    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute("SET geqo_threshold = {}".format(32))
    set_cost_model(cursor, cost_model)

    if materialize:
        cursor.execute("SET enable_material = on")

    # the idea is that we inject the #rows using the pg_hint_plan into
    # postgresql; then the join orders etc. are computed the postgresql engine
    # using the given cardinalities
    cursor.execute("SET join_collapse_limit = {}".format(32))
    cursor.execute("SET from_collapse_limit = {}".format(32))

    if timeout is not None:
        cursor.execute("SET statement_timeout = {}".format(timeout))

    start = time.time()

    try:
        cursor.execute(sql)
    except KeyboardInterrupt:
        print("killed because of ctrl+c")
        sys.exit(-1)
    except Exception as e:
        print(e)
        # cursor.execute("ROLLBACK")
        # con.commit()
        if not "timeout" in str(e):
            print("failed to execute for reason other than timeout")
            print(e)
            # print(sql)
            # cursor.close()
            con.close()
            time.sleep(6)
            return None, -1.0
        else:
            print("failed because of timeout!")
            end = time.time()

            cursor.close()
            con.close()
            return None, (timeout/1000) + 9.0

    explain_output = cursor.fetchall()
    end = time.time()

    # print("took {} seconds".format(end-start))
    # sys.stdout.flush()
    cursor.close()
    con.close()

    return explain_output, end-start

def main():
    def add_runtime_row(qname, rt, exp_analyze):
        cur_runtimes["qname"].append(qname)
        cur_runtimes["runtime"].append(rt)
        cur_runtimes["exp_analyze"].append(exp_analyze)

    cost_model = args.cost_model
    costs_fn = args.costs_fn_tmp.format(args.cost_model)
    costs_fn = os.path.join(args.result_dir, costs_fn)

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
    total_rt = 0.0

    if args.samples_type is not None:
        if args.samples_type == "debug":
            costs = costs[costs.qname.str.len() <= 7]
            costs = costs[~costs.qname.str.contains("6a")]
            costs = costs[~costs.qname.str.contains("7a")]
            costs = costs[~costs.qname.str.contains("2b")]
            costs = costs[~costs.qname.str.contains("2a")]
            costs = costs[~costs.qname.str.contains("2c")]
        else:
            costs = costs[costs.samples_type.str.contains(args.samples_type)]


    # print(set(costs["samples_type"]))
    # pdb.set_trace()
    print("Going to execute {} queries, {} reps each".format(
        len(costs), args.reps))

    for rep in range(args.reps):
        costs = costs.sample(frac=1.0)
        for i,row in costs.iterrows():
            # if "samples_type" in list(row.keys()) and \
                    # args.samples_type is not None:
                # if args.samples_type not in row["samples_type"]:
                    # continue

            if row["qname"] in runtimes["qname"].values:
                # what is the stored value for this key?
                rt_df = runtimes[runtimes["qname"] == row["qname"]]
                stored_rt = rt_df["runtime"].values[0]
                if stored_rt == TIMEOUT_CONSTANT and args.rerun_timeouts:
                    print("going to rerun timed out query")
                elif stored_rt == -1 and args.rerun_timeouts:
                    print("going to rerun failed query")
                else:
                    pass
                    # print("skipping {} with stored runtime".format(row["qname"]))
                    # continue

            exp_analyze, rt = execute_sql(row["exec_sql"],
                    cost_model=cost_model,
                    explain=args.explain,
                    timeout=args.timeout,
                    materialize = args.materialize,
                    drop_cache=args.drop_cache)

            if rt >= 0.0:
                total_rt += rt

            add_runtime_row(row["qname"], rt, exp_analyze)

            # cur_runtimes = pd.DataFrame(cur_runtimes)
            # tmp = cur_runtimes[cur_runtimes["runtime"] >= 0.0]

            rts = np.array(cur_runtimes["runtime"])
            rts = rts[rts >= 0.0]
            num_fails = len(cur_runtimes["runtime"]) - len(rts)

            print("{}, Rep: {}, Cur: {}, CurRT: {}, TotalRT: {}, #Queries:{}, AvgRt: {}, #Fails: {}"\
                .format(
                args.result_dir,
                rep,
                row["qname"], rts[-1],
                round(total_rt,2), len(rts),
                round(sum(rts) / len(rts), 2), num_fails))
            sys.stdout.flush()

            df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
            df.to_csv(rt_fn, index=False)

    print("Total runtime was: ", total_rt)

if __name__ == "__main__":
    args = read_flags()
    main()
