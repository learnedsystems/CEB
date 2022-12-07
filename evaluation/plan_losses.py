import psycopg2 as pg
import getpass
import numpy as np
from query_representation.utils import *
from .cost_model import *
import multiprocessing as mp
import math

import pdb
import klepto
import copy

## for using pg_hint_plan; Refer to their documentation for more details.
PG_HINT_CMNT_TMP = '''/*+ {COMMENT} */'''
PG_HINT_JOIN_TMP = "{JOIN_TYPE}({TABLES}) "
PG_HINT_CARD_TMP = "Rows({TABLES} #{CARD}) "
PG_HINT_SCAN_TMP = "{SCAN_TYPE}({TABLE}) "
PG_HINT_LEADING_TMP = "Leading({JOIN_ORDER})"
PG_HINT_JOINS = {}
PG_HINT_JOINS["Nested Loop"] = "NestLoop"
PG_HINT_JOINS["Hash Join"] = "HashJoin"
PG_HINT_JOINS["Merge Join"] = "MergeJoin"

PG_HINT_SCANS = {}
PG_HINT_SCANS["Seq Scan"] = "SeqScan"
PG_HINT_SCANS["Index Scan"] = "IndexScan"
PG_HINT_SCANS["Index Only Scan"] = "IndexOnlyScan"
PG_HINT_SCANS["Bitmap Heap Scan"] = "BitmapScan"
PG_HINT_SCANS["Tid Scan"] = "TidScan"

def get_pg_cost_from_sql(sql, cur):
    assert "explain" in sql
    cur.execute(sql)
    explain = cur.fetchall()
    all_costs = extract_values(explain[0][0][0], "Total Cost")
    mcost = max(all_costs)
    cost = explain[0][0][0]["Plan"]["Total Cost"]
    return mcost, explain

def _gen_pg_hint_cards(cards):
    '''
    '''
    card_str = ""
    for aliases, card in cards.items():
        if isinstance(aliases, tuple):
            aliases = " ".join(aliases)
        card_line = PG_HINT_CARD_TMP.format(TABLES = aliases,
                                            CARD = card)
        card_str += card_line
    return card_str

def _gen_pg_hint_join(join_ops):
    '''
    '''
    join_str = ""
    for tables, join_op in join_ops.items():
        join_line = PG_HINT_JOIN_TMP.format(TABLES = tables,
                                            JOIN_TYPE = PG_HINT_JOINS[join_op])
        join_str += join_line
    return join_str

def _gen_pg_hint_scan(scan_ops):
    '''
    '''
    scan_str = ""
    for alias, scan_op in scan_ops.items():
        scan_line = PG_HINT_SCAN_TMP.format(TABLE = alias,
                                            SCAN_TYPE = PG_HINT_SCANS[scan_op])
        scan_str += scan_line
    return scan_str

def get_leading_hint(join_graph, explain):
    '''
    Ryan's implementation.
    '''
    def __extract_jo(plan):
        if plan["Node Type"] in join_types:
            left = list(extract_aliases(plan["Plans"][0], jg=join_graph))
            right = list(extract_aliases(plan["Plans"][1], jg=join_graph))

            if len(left) == 1 and len(right) == 1:
                left_alias = left[0][left[0].lower().find(" as ")+4:]
                right_alias = right[0][right[0].lower().find(" as ")+4:]
                return left_alias +  " " + right_alias

            if len(left) == 1:
                left_alias = left[0][left[0].lower().find(" as ")+4:]
                return left_alias + " (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                right_alias = right[0][right[0].lower().find(" as ")+4:]
                return "(" + __extract_jo(plan["Plans"][0]) + ") " + right_alias

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    jo = __extract_jo(explain[0][0][0]["Plan"])
    jo = "(" + jo + ")"
    return PG_HINT_LEADING_TMP.format(JOIN_ORDER = jo)


def get_pghint_modified_sql(sql, cardinalities, join_ops,
        leading_hint, scan_ops):
    '''
    @cardinalities: dict
    @join_ops: dict

    @ret: sql, augmented with appropriate comments.
    '''
    if "explain (format json)" not in sql:
        sql = " explain (format json) " + sql

    comment_str = ""
    if cardinalities is not None:
        card_str = _gen_pg_hint_cards(cardinalities)
        # gen appropriate sql with comments etc.
        comment_str += card_str

    if join_ops is not None:
        join_str = _gen_pg_hint_join(join_ops)
        comment_str += join_str + " "
    if leading_hint is not None:
        comment_str += leading_hint + " "
    if scan_ops is not None:
        scan_str = _gen_pg_hint_scan(scan_ops)
        comment_str += scan_str + " "

    pg_hint_str = PG_HINT_CMNT_TMP.format(COMMENT=comment_str)
    sql = pg_hint_str + sql
    return sql

def _get_pg_plancost(query, est_cardinalities,
        true_cardinalities,
        join_graph, cursor, sql_costs):
    '''
    Main function for computing Postgres Plan Costs.
    '''
    est_card_sql = get_pghint_modified_sql(query, est_cardinalities, None,
            None, None)
    assert "explain" in est_card_sql.lower()

    cursor.execute(est_card_sql)
    explain = cursor.fetchall()

    # print(join_graph.nodes(data=True))
    est_join_order_sql, est_join_ops, scan_ops = get_pg_join_order(join_graph,
            explain)
    leading_hint = get_leading_hint(join_graph, explain)

    # print(est_join_order_sql)
    # print(leading_hint)

    est_opt_sql = nx_graph_to_query(join_graph,
            from_clause=est_join_order_sql)

    # if "COUNT(*)" in est_opt_sql:
        # print(est_opt_sql[0:100])
        # print("COUNT* in est_opt_sql!")
        # pdb.set_trace()
    # return "", 0.0, ""

    # add the join ops etc. information
    cost_sql = get_pghint_modified_sql(est_opt_sql, true_cardinalities,
            est_join_ops, leading_hint, scan_ops)

    ## original code
    # est_cost, est_explain = get_pg_cost_from_sql(cost_sql, cursor)

    # cost_sql will be seen often, as true_cardinalities remain fixed; and
    # different estimates can lead to the same plan. So we cache the results.
    # Notice that cost_sql involves setting up all the true estimates, join
    # ops, and other pg hints  in the sql string -- thus if the exact same
    # string is repeated, then its PostgreSQL cost would be the same too.
    cost_sql_key = deterministic_hash(cost_sql)

    if sql_costs is not None:
        if cost_sql_key in sql_costs.archive:
            try:
                est_cost, est_explain = sql_costs.archive[cost_sql_key]
            except:
                # just the default in case bad things happened
                est_cost, est_explain = get_pg_cost_from_sql(cost_sql, cursor)
                sql_costs.archive[cost_sql_key] = (est_cost, est_explain)
        else:
            est_cost, est_explain = get_pg_cost_from_sql(cost_sql, cursor)
            sql_costs.archive[cost_sql_key] = (est_cost, est_explain)
    else:
        # without caching
        est_cost, est_explain = get_pg_cost_from_sql(cost_sql, cursor)
        leading_hint2 = get_leading_hint(join_graph, est_explain)

        # if leading_hint != leading_hint2:
            # print("leading hint NOT matches!")
            # assert False

    # set this to sql to be executed, as pg_hint_plan will enforce the
    # estimated cardinalities, and let postgres make decisions for join order
    # and everything about operators based on the estimated cardinalities
    exec_sql = get_pghint_modified_sql(est_opt_sql, est_cardinalities,
            None, None, None)

    return exec_sql, est_cost, est_explain

def compute_cost_pg_single(queries, join_graphs, true_cardinalities,
        est_cardinalities, opt_costs, user, pwd, db_host, port, db_name,
        use_qplan_cache, cost_model):
    '''
    Just a wrapper function around the PPC methods --- separate
    function so we can call it using multiprocessing. See
    PPC.compute_cost method for most argument descriptions.

    @use_qplan_cache: query plans for the same query can be repeated often;
    Setting this to true uses a cache across runs for such plans.
    '''
    # some weird effects between different installations
    try:
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd, host=db_host)
    except:
        print("connection failed")

        pdb.set_trace()
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd)

    # can take a lot of space
    # if use_qplan_cache:
    if False:
        archive_fn = "./.lc_cache/sql_costs_" + cost_model
        sql_costs_archive = klepto.archives.dir_archive(archive_fn,
                cached=True, serialized=True)
    else:
        sql_costs_archive = None

    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    set_cost_model(cursor, cost_model)

    ret = []
    for i, query in enumerate(queries):
        join_graph = join_graphs[i]
        est_sql, est_cost, est_explain = _get_pg_plancost(query,
                est_cardinalities[i], true_cardinalities[i], join_graphs[i],
                cursor, sql_costs_archive)

        if opt_costs[i] is None:
            _, opt_costs[i], _ = _get_pg_plancost(query,
                    true_cardinalities[i], true_cardinalities[i],
                    join_graphs[i], cursor, sql_costs_archive)
            if est_cost < opt_costs[i]:
                est_cost = opt_costs[i]

        ret.append((est_cost, opt_costs[i], est_explain,
            est_sql))

    cursor.close()
    con.close()
    return ret

class PPC():

    def __init__(self, cost_model, user, pwd, db_host, port, db_name):
        '''
        @cost_model: str.
        '''
        self.cost_model = cost_model
        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name

        opt_archive_fn = "./.lc_cache/opt_archive_" + cost_model
        self.opt_archive = klepto.archives.dir_archive(opt_archive_fn,
                cached=True, serialized=True)

    def compute_costs(self, sqls, join_graphs, true_cardinalities,
            est_cardinalities, num_processes=8,
            use_qplan_cache=False,
            pool=None):
        '''
        @query_dict: [sqls]
        @true_cardinalities / est_cardinalities: [{}]
                dictionary, specifying cardinality of each subplan
                key: str; "alias1 alias2 ... aliasN" for the N tables in the
                subplan, in the order they were stored in the qrep object.
                val: cardinality (double)
        @backend: only supports postgres for now.
        @pool: multiprocessing pool, if None, just compute it in a single thread.
        @ret:
            costs: [cost1, ..., ] true costs (PPC) of the plans generated using
            the estimated cardinalities.
            opt_costs: [cost1 ...] costs computed using true cardinalities.
            Useful, if you want to consider other metrics than PPC -- such as
            the ratio of the estimated / optimal cost; Or the difference etc.

            explains: postgres explains from which the costs were computed.
            exec_sqls: these are sqls with appropriate pghint hints set up to
            force the join order / operators / scan ops computed using the
            estimated cardinalities. These can be executed on postgresql with
            pghint plugin set up.
        '''
        start = time.time()
        assert isinstance(sqls, list)
        assert isinstance(true_cardinalities, list)
        assert isinstance(est_cardinalities, list)
        assert len(sqls) == len(true_cardinalities) == len(est_cardinalities)

        # declaring return arrays
        est_costs = [None]*len(sqls)
        opt_costs = [None]*len(sqls)
        est_explains = [None]*len(sqls)
        est_sqls = [None]*len(sqls)

        # opt_cost is the cost using true cardinalities, thus they are constant
        # for a given query + cost model.
        if use_qplan_cache:
            for i, sql in enumerate(sqls):
                sql_key = deterministic_hash(sql + self.cost_model)
                if sql_key in self.opt_archive.archive:
                    opt_costs[i] = self.opt_archive.archive[sql_key]

        if pool is None:
            # single threaded case, useful for debugging
            # assert False
            all_costs = [compute_cost_pg_single(sqls, join_graphs,
                    true_cardinalities, est_cardinalities, opt_costs,
                    self.user,
                    self.pwd, self.db_host, self.port, self.db_name, False,
                    self.cost_model)]
            batch_size = len(sqls)
        else:
            num_processes = pool._processes
            batch_size = max(1, math.ceil(len(sqls) / num_processes))
            assert num_processes * batch_size >= len(sqls)
            par_args = []
            for proc_num in range(num_processes):
                start_idx = proc_num * batch_size
                end_idx = min(start_idx + batch_size, len(sqls))
                par_args.append((sqls[start_idx:end_idx],
                    join_graphs[start_idx:end_idx],
                    true_cardinalities[start_idx:end_idx],
                    est_cardinalities[start_idx:end_idx],
                    opt_costs[start_idx:end_idx],
                    self.user, self.pwd, self.db_host,
                    self.port, self.db_name, use_qplan_cache, self.cost_model))

            # pdb.set_trace()
            all_costs = pool.starmap(compute_cost_pg_single, par_args)

        new_seen = False
        for num_proc, costs in enumerate(all_costs):
            start_idx = int(num_proc * batch_size)
            for i, (est, opt, est_explain, est_sql) \
                        in enumerate(costs):
                est_costs[start_idx+i] = est
                est_explains[start_idx+i] = est_explain
                est_sqls[start_idx+i] = est_sql
                opt_costs[start_idx+i] = opt
                # pool is None used only in special cases / debugging
                sql = sqls[start_idx + i]
                sql_key = deterministic_hash(sql)
                if sql_key not in self.opt_archive.archive \
                        and pool is not None:
                    self.opt_archive.archive[sql_key] = opt

        return np.array(est_costs), np.array(opt_costs), est_explains, est_sqls

def get_shortest_path_costs(samples, source_node,
        all_ests, all_trues, cost_model):
    '''
    @ret: cost of the given path in subsetg.
    '''
    cost_key = "est_cost"
    costs = []
    opt_costs = []
    paths = []

    assert all_trues is not None
    assert all_ests is not None

    assert len(all_trues) == len(all_ests)

    for i in range(len(samples)):
        # subsetg = samples[i]["subset_graph_paths"]
        ## TODO: we should not need to recompute the costs here
        subsetg = samples[i]["subset_graph"]
        assert source_node in subsetg.nodes()

        ests = all_ests[i]
        update_subplan_costs(subsetg, cost_model, cost_key=cost_key,
                ests=ests)

        true_costs_known = False
        # TODO: only need this if it doesn't already exist; check to make sure
        # this works
        # for edge in subsetg.edges():
            # if cost_model + "cost" in subsetg[edge[0]][edge[1]].keys():
                # true_costs_known = True
            # break

        if not true_costs_known:
            trues = all_trues[i]
            update_subplan_costs(subsetg, cost_model, cost_key="cost",
                    ests=trues)

        # TODO: can precompute final node for each query
        nodes = list(subsetg.nodes())
        nodes.sort(key=lambda x: len(x))
        final_node = nodes[-1]

        # TODO: can cache this
        opt_path = nx.shortest_path(subsetg, final_node,
                source_node, weight=cost_model+"cost")[0:-1]
        path = nx.shortest_path(subsetg, final_node,
                source_node, weight=cost_model+cost_key)[0:-1]
        paths.append(path)

        opt_cost = 0.0
        cost = 0.0
        scan_types = {}
        for pi in range(len(path)-1):
            true_cost_key = cost_model + "cost"
            scan_key = true_cost_key + "scan_type"
            cost += subsetg[path[pi]][path[pi+1]][true_cost_key]
            opt_cost += subsetg[opt_path[pi]][opt_path[pi+1]][true_cost_key]

            if scan_key in subsetg[path[pi]][path[pi+1]]:
                scan_types.update(subsetg[path[pi]][path[pi+1]][scan_key])

        assert cost >= 1
        costs.append(cost)
        opt_costs.append(opt_cost)

    return costs, opt_costs, paths

class PlanCost():
    def __init__(self, cost_model):
        '''
        @cost_model: str.
        '''
        self.cost_model = cost_model

    def compute_costs(self, qreps, ests, pool=None):
        '''
        @ests: [dicts] of estimates
        '''
        start = time.time()
        subsetgs = []

        true_cardinalities = []
        for i, qrep in enumerate(qreps):
            trues = {}
            for node, node_info in qrep["subset_graph"].nodes().items():
                trues[node] = node_info["cardinality"]["actual"]
            true_cardinalities.append(trues)

        for qrep in qreps:
            qkey = deterministic_hash(qrep["sql"])
            # update qreps to include SOURCE node so we can run shortest path
            # functions on it
            subsetg = qrep["subset_graph"]
            add_single_node_edges(subsetg, SOURCE_NODE)

        num_processes = pool._processes
        batch_size = max(1, math.ceil(len(qreps) / num_processes))
        assert num_processes * batch_size >= len(qreps)
        par_args = []

        for proc_num in range(num_processes):
            start_idx = proc_num * batch_size
            end_idx = min(start_idx + batch_size, len(qreps))
            if end_idx <= start_idx:
                continue
            par_args.append((qreps[start_idx:end_idx],
                SOURCE_NODE,
                ests[start_idx:end_idx],
                true_cardinalities[start_idx:end_idx],
                self.cost_model))

        all_opt_costs = []
        all_costs = []
        all_costs_batched = pool.starmap(get_shortest_path_costs,
                par_args)

        for c in all_costs_batched:
            all_costs += c[0]
            all_opt_costs += c[1]
            # we are ignoring the paths (c[1]) for now; but those can be used
            # to analyze the common plans; or inject those plans into postgres
            # etc.

        # remove the ndoe we added temporarily
        for qrep in qreps:
            qrep["subset_graph"].remove_node(SOURCE_NODE)

        return np.array(all_costs), np.array(all_opt_costs)
