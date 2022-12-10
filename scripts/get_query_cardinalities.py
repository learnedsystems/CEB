import sys
sys.path.append(".")
import argparse
import psycopg2 as pg
import glob

from query_representation.utils import *
from query_representation.query import *

import pdb
import random
import klepto
from multiprocessing import Pool, cpu_count
import json
import pickle

from networkx.readwrite import json_graph
import re
from query_representation.query import parse_sql

from wanderjoin import WanderJoin
import math

import scipy.stats as st

TIMEOUT_COUNT_CONSTANT = 150001000001
CROSS_JOIN_CONSTANT = 150001000000
EXCEPTION_COUNT_CONSTANT = 150001000002
RERUN_TIMEOUTS = 1

CACHE_TIMEOUT = 4
CACHE_CARD_TYPES = ["actual"]

DEBUG_CHECK_TIMES = False
CONF_ALPHA = 0.99

def pg_est_from_explain(output):
    '''
    '''
    est_vals = None
    for line in output:
        line = line[0]
        # getting estimate from the first Join, or Scan operator we see
        if "Seq Scan" in line or "Loop" in line or "Join" in line \
                or "Index Scan" in line or "Scan" in line:
            for w in line.split():
                if "rows" in w and est_vals is None:
                    est_vals = int(re.findall("\d+", w)[0])
                    return est_vals

    print("pg est failed!")
    print(output)
    pdb.set_trace()
    return 1.00

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="ceb")
    parser.add_argument("--pwd", type=str, required=False,
            default="password")

    parser.add_argument("--card_cache_dir", type=str, required=False,
            default="./cardinality_cache")
    parser.add_argument("--port", type=str, required=False,
            default=5432)
    parser.add_argument("--wj_walk_timeout", type=float, required=False,
            default=0.5)
    parser.add_argument("--query_dir", type=str, required=False,
            default=None)
    parser.add_argument("-n", "--num_queries", type=int,
            required=False, default=-1)
    parser.add_argument("--use_tries", type=int,
            required=False, default=1)
    parser.add_argument("--skip_zero_queries", type=int,
            required=False, default=1)
    parser.add_argument("--no_parallel", type=int,
            required=False, default=0)
    parser.add_argument("--card_type", type=str, required=False,
            default=None)
    parser.add_argument("--key_name", type=str, required=False,
            default=None)
    parser.add_argument("--true_timeout", type=int,
            required=False, default=1800000*5)
    parser.add_argument("--pg_total", type=int,
            required=False, default=1)
    parser.add_argument("--num_proc", type=int,
            required=False, default=-1)
    parser.add_argument("--seed", type=int,
            required=False, default=1234)
    parser.add_argument("--sampling_percentage", type=int,
            required=False, default=None)
    parser.add_argument("--sampling_type", type=str,
            required=False, default=None)
    parser.add_argument("--db_year", type=int,
            required=False, default=None)

    return parser.parse_args()

def update_bad_qrep(qrep):
    qrep = parse_sql(qrep["sql"], None, None, None, None, None,
            compute_ground_truth=False)
    qrep["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
    qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])
    return qrep

def is_cross_join(sg):
    '''
    enforces the constraint that the graph should be connected.
    '''
    if len(sg.nodes()) < 2:
        # FIXME: should be return False
        return False
    sg2 = nx.Graph(sg)
    to_remove = []

    # do this in case of stackexchange database; because of the weird query
    # structure, if the graph is connected only through the `site` table, it
    # still behaves like a cross-join. Check the appendix of the Flow-Loss
    # paper for more details
    for node, data in sg2.nodes(data=True):
        if data["real_name"] == "site":
            to_remove.append(node)

    for node in to_remove:
        sg2.remove_node(node)
    if nx.is_connected(sg2):
        return False
    return True

def get_cardinality_wj(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, fn, wj_fn, wj_walk_timeout, idx, seed, trie_cache, use_tries):

    key_name = "wanderjoin-" + str(wj_walk_timeout)
    # key_name = "wj" + str(wj_walk_timeout)
    for subset, info in qrep["subset_graph"].nodes().items():
        cards = info["cardinality"]
        if key_name in cards:
            return
        if "actual" not in cards:
            return

    if idx % 10 == 0:
        print("query: ", idx)
    start = time.time()
    wj = WanderJoin(user, pwd, db_host, port,
            db_name, verbose=True, walks_timeout=wj_walk_timeout, seed =
            seed, use_tries=use_tries, trie_cache=trie_cache)

    if SOURCE_NODE in list(qrep["subset_graph"].nodes()):
        qrep["subset_graph"].remove_node(SOURCE_NODE)

    data = wj.get_counts(qrep)

    # save wj data
    for subset, info in qrep["subset_graph"].nodes().items():
        cards = info["cardinality"]
        num = data["card_samples"][subset]
        est = math.ceil(data["card_ests_sum"][subset] / num)
        if num <= 1:
            std = 0
        else:
            std = np.sqrt(data["card_vars_sum"][subset] / float(num-1))
        alpha = st.norm.ppf((CONF_ALPHA+1)/2)
        half_interval = std*alpha / np.sqrt(num)

        cards[key_name] = est
        cards[key_name + "_half_interval"] = half_interval

    old_data = load_object(wj_fn)
    if old_data is None:
        old_data = {}
    old_data[key_name] = data

    save_qrep(fn, qrep)
    save_object(wj_fn, old_data)
    print("wanderjoin, seed: {} took {}".format(seed, time.time()-start))
    return qrep


def get_cardinality(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, true_timeout, pg_total, cache_dir, fn, wj_walk_timeout, idx,
        sampling_percentage, sampling_type, skip_zero_queries, db_year):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    print("get cardinality!")
    if key_name is None:
        key_name = card_type

    if db_year is not None:
        db_name = db_name + str(db_year)

    if sampling_percentage is not None:
        key_name = str(sampling_type) + str(sampling_percentage) + "_" + key_name

        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)

        cursor = con.cursor()

    if idx % 10 == 0:
        print("query: ", idx)

    # load the cache for few types
    # if card_type in CACHE_CARD_TYPES:
        # sql_cache = klepto.archives.dir_archive(cache_dir,
                # cached=True, serialized=True)

    found_in_cache = 0
    existing = 0
    num_timeout = 0
    site_cj = 0
    query_exec_times = []

    node_list = list(qrep["subset_graph"].nodes())
    node_list.sort(reverse=True, key = lambda x: len(x))
    if args.db_name == "so":
        source_node = tuple(["SOURCE"])
        if source_node in node_list:
            node_list.remove(source_node)
    elif args.db_name == "imdb":
        source_node = tuple(["s"])
        if source_node in node_list:
            node_list.remove(source_node)

    card_key = "cardinality"
    if db_year is not None:
        card_key = str(db_year) + card_key

    for subqi, subset in enumerate(node_list):

        info = qrep["subset_graph"].nodes()[subset]
        if card_key not in info:
            info[card_key] = {}

        if "exec_time" not in info:
            info["exec_time"] = {}

        cards = info[card_key]
        execs = info["exec_time"]
        sg = qrep["join_graph"].subgraph(subset)
        subsql = nx_graph_to_query(sg)

        if sampling_percentage is not None:
            table_names = []
            for k,v in sg.nodes(data=True):
                table = v["real_name"]
                new_table_name = table + "_" + sampling_type + str(sampling_percentage)
                new_table_name += " "
                new_table_name = " " + new_table_name
                # TODO: check if table exists...
                cursor.execute("select * from information_schema.tables where table_name='{}'".format(new_table_name))
                if bool(cursor.rowcount):
                    subsql = re.sub(r"\b {} \b".format(table), new_table_name,
                            subsql)

        if key_name in cards \
                and not DEBUG_CHECK_TIMES:
            if key_name == "actual":
                if cards[key_name] == 0 and skip_zero_queries:
                    # don't want to get cardinalities for zero queries
                    break

                elif cards[key_name] >= TIMEOUT_COUNT_CONSTANT and not RERUN_TIMEOUTS:
                    existing += 1
                    continue

                elif cards[key_name] == EXCEPTION_COUNT_CONSTANT:
                    existing += 1
                    continue

                elif cards[key_name] < TIMEOUT_COUNT_CONSTANT:
                    existing += 1
                    continue

                print("key existing: {}, but going to rerun".format(cards[key_name]))

            # TODO: not sure why this here
            # if not (sampling_percentage is not None and \
                    # cards[key_name] >= TIMEOUT_COUNT_CONSTANT):
                # existing += 1
                # continue


        if card_type == "pg":
            subsql = "EXPLAIN " + subsql
            output = execute_query(subsql, user, db_host, port, pwd, db_name, [])
            card = pg_est_from_explain(output)
            cards[key_name] = card
            if subqi % 10 == 0:
                save_qrep(fn, qrep)

        elif card_type == "actual":
            if subqi % 10 == 0:
                save_qrep(fn, qrep)

            hash_sql = deterministic_hash(subsql)
            if "count" not in subsql.lower():
                print("cardinality query does not have count")
                pdb.set_trace()
            if is_cross_join(sg):
                site_cj += 1
                card = CROSS_JOIN_CONSTANT
                cards[key_name] = card
                continue

            if hash_sql in sql_cache.archive \
                    and not DEBUG_CHECK_TIMES:
                card = sql_cache.archive[hash_sql]
                found_in_cache += 1
                cards[key_name] = card
                continue

            start = time.time()
            pre_execs = ["SET statement_timeout = {}".format(true_timeout)]
            output = execute_query(subsql, user, db_host, port, pwd, db_name,
                            pre_execs)
            if isinstance(output, Exception):
                print(output)
                card = EXCEPTION_COUNT_CONSTANT
                num_timeout += 1
                # continue
                # pdb.set_trace()
            elif output == "timeout":
                print("timeout query: ")
                print(subsql)
                card = TIMEOUT_COUNT_CONSTANT
                num_timeout += 1
            else:
                card = output[0][0]

            exec_time = time.time() - start
            if exec_time > CACHE_TIMEOUT:
                print(exec_time)
                sql_cache.archive[hash_sql] = card
            cards[key_name] = card
            execs[key_name] = exec_time
            query_exec_times.append(exec_time)
            if card == 0 and skip_zero_queries:
                # bad times...
                print("skipping query with zero cardinality subquery")
                break

        elif card_type == "wanderjoin":
            assert "SELECT" in subsql
            subsql = subsql.replace("SELECT", "SELECT ONLINE")
            subsql = subsql.replace(";","")
            subsql += WANDERJOIN_TIME_FMT.format(
                    TIME = wj_walk_timeout,
                    CONF = 95,
                    INT = 1000)
            print(subsql)
            output = execute_query(subsql, user, db_host, port, pwd, db_name,
                            [])
            print(output)
            pdb.set_trace()
            assert False

        elif card_type == "total":
            exec_sql = get_total_count_query(subsql)
            if args.pg_total:
                exec_sql = "EXPLAIN " + exec_sql

            output = execute_query(exec_sql, user, db_host, port, pwd, db_name, [])
            card = pg_est_from_explain(output)
            cards[key_name] = card
        else:
            assert False

    if card_type == "actual":
        print("total: {}, timeout: {}, existing: {}, found in cache: {}".format(\
                len(qrep["subset_graph"].nodes()), num_timeout, existing, found_in_cache))
        # print("site cj: ", site_cj)
        if len(query_exec_times) != 0:
            print("avg exec time: ", sum(query_exec_times) / len(query_exec_times))

    if fn is not None:
        # update_qrep(qrep)
        save_qrep(fn, qrep)
        print("updated sql rep!")

    sys.stdout.flush()
    return qrep

def main():
    fns = list(glob.glob(args.query_dir + "/*"))
    fns.sort()
    par_args = []

    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break

        if (".pkl" not in fn and ".sql" not in fn):
            continue

        if ".pkl" in fn:
            qrep = load_qrep(fn)
        else:
            with open(fn, "r") as f:
                sql = f.read()
            sql = sql.strip()
            if "SELECT" not in sql:
                continue

            qrep = parse_sql(sql, None, None, None, None, None,
                    compute_ground_truth=False)

            qrep["subset_graph"] = \
                    nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
            qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])
            fn = fn.replace(".sql", ".pkl")
            save_qrep(fn, qrep)
            print("updated sql rep!")
            continue

        if args.no_parallel:
            if args.card_type == "wanderjoin":
                wj_dir = os.path.dirname(fn) + "/wj_data/"
                base_name = os.path.basename(fn)
                if not os.path.exists(wj_dir):
                    make_dir(wj_dir)
                wj_fn = wj_dir + base_name
                get_cardinality_wj(qrep, args.card_type, args.key_name, args.db_host,
                        args.db_name, args.user, args.pwd, args.port,
                         fn, wj_fn, args.wj_walk_timeout, i, args.seed, None,
                         args.use_tries)
                print("done!")
                pdb.set_trace()
            else:
                get_cardinality(qrep, args.card_type, args.key_name, args.db_host,
                        args.db_name, args.user, args.pwd, args.port,
                        args.true_timeout, args.pg_total, args.card_cache_dir, fn,
                        args.wj_walk_timeout, i, args.sampling_percentage,
                        args.sampling_type, True, args.db_year)

            continue

        if args.card_type == "wanderjoin":
            par_func = get_cardinality_wj

            wj_dir = os.path.dirname(fn) + "/wj_data/"
            base_name = os.path.basename(fn)
            if not os.path.exists(wj_dir):
                make_dir(wj_dir)
            wj_fn = wj_dir + base_name
            # trie_cache = klepto.archives.dir_archive("./trie_cache",
                    # cached=True, serialized=True)
            # tstart = time.time()
            # print("going to load trie archive...")
            # trie_cache.load()
            # print("loading trie archive took: ", time.time() - tstart)
            par_args.append((qrep, args.card_type, args.key_name, args.db_host,
                    args.db_name, args.user, args.pwd, args.port,
                     fn, wj_fn, args.wj_walk_timeout, i, args.seed, None,
                     args.use_tries))
        else:
            par_func = get_cardinality
            par_args.append((qrep, args.card_type, args.key_name, args.db_host,
                    args.db_name, args.user, args.pwd, args.port,
                    args.true_timeout, args.pg_total, args.card_cache_dir, fn,
                    args.wj_walk_timeout, i, args.sampling_percentage,
                    args.sampling_type, args.skip_zero_queries, args.db_year))

    if args.no_parallel:
        print("Generated all cardinalities")
        exit(1)

    start = time.time()
    if args.num_proc == -1:
        num_proc = cpu_count()
    else:
        num_proc = args.num_proc
    with Pool(processes = num_proc) as pool:
        qreps = pool.starmap(par_func, par_args)
    print("Generated all cardinalities in {} seconds".format(time.time()-start))

args = read_flags()
main()
