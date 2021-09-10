import networkx as nx
from networkx.readwrite import json_graph

from query_representation.utils import *
import time
import itertools
import json
import pdb
import pickle
import copy

def get_subset_cache_name(sql):
    return str(deterministic_hash(sql)[0:5])

def parse_sql(sql, user, db_name, db_host, port, pwd, timeout=False,
        compute_ground_truth=True, subset_cache_dir="./subset_cache/"):
    '''
    @sql: sql query string.

    @ret: python dict with the keys:
        sql: original sql string
        join_graph: networkX graph representing query and its
        join_edges. Properties include:
            Nodes:
                - table
                - alias
                - predicate matches
            Edges:
                - join_condition

            Note: This is the only place where these strings will be stored.
            Each of the subplans will be represented by their nodes within
            the join_graph, and we can use these properties to reconstruct the
            appropriate query for each subplan.

        subset_graph: networkX graph representing each subplan as a node.

        Properties of each subplan will include all the cardinality data that
        will need to be computed:
            - true_count
            - pg_count
            - total_count
    '''
    start = time.time()
    join_graph = extract_join_graph(sql)
    subset_graph = generate_subset_graph(join_graph)

    print("query has",
          len(join_graph.nodes), "relations,",
          len(join_graph.edges), "joins, and",
          len(subset_graph), " possible subplans.",
          "took:", time.time() - start)

    ret = {}
    ret["sql"] = sql
    ret["join_graph"] = join_graph
    ret["subset_graph"] = subset_graph

    ret["join_graph"] = nx.adjacency_data(ret["join_graph"])
    ret["subset_graph"] = nx.adjacency_data(ret["subset_graph"])
    return ret

def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)

    query["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])

    return query

def save_qrep(fn, cur_qrep):
    assert ".pkl" in fn
    qrep = copy.deepcopy(cur_qrep)
    qrep["join_graph"] = nx.adjacency_data(qrep["join_graph"])
    qrep["subset_graph"] = nx.adjacency_data(qrep["subset_graph"])

    with open(fn, "wb") as f:
        pickle.dump(qrep, f)

def get_tables(qrep):
    '''
    ret:
        @tables: list of table names in the query
        @aliases: list of corresponding aliases in the query.
        (each table has an alias here.)
    '''
    tables = []
    aliases = []
    for node in qrep["join_graph"].nodes(data=True):
        aliases.append(node[0])
        tables.append(node[1]["real_name"])

    return tables, aliases

def get_predicates(qrep):
    '''
    ret:
        @predicates: list of the predicate strings in the query
        We also break the each predicate string into @pred_cols, @pred_types,
        and @pred_vals and return those as separate lists.
    '''
    predicates = []
    pred_cols = []
    pred_types = []
    pred_vals = []
    for node in qrep["join_graph"].nodes(data=True):
        info = node[1]
        if len(info["predicates"]) == 0:
            continue
        predicates.append(info["predicates"])
        pred_cols.append(info["pred_cols"])
        pred_types.append(info["pred_types"])
        pred_vals.append(info["pred_vals"])

    return predicates, pred_cols, pred_types, pred_vals

def get_joins(qrep):
    '''
    '''
    joins = []
    for einfo in qrep["join_graph"].edges(data=True):
        join = einfo[2]["join_condition"]
        joins.append(join)
    return joins

def get_postgres_cardinalities(qrep):
    '''
    @ests: dict; key: label of the subplan. value: cardinality estimate.
    '''
    pred_dict = {}
    for alias_key in qrep["subset_graph"].nodes():
        info = qrep["subset_graph"].nodes()[alias_key]
        est = info["cardinality"]["expected"]
        pred_dict[(alias_key)] = est

    return pred_dict

def get_true_cardinalities(qrep):
    '''
    @ests: dict; key: label of the subplan. value: cardinality estimate.
    '''
    pred_dict = {}
    for alias_key in qrep["subset_graph"].nodes():
        info = qrep["subset_graph"].nodes()[alias_key]
        true_card = info["cardinality"]["actual"]
        pred_dict[(alias_key)] = true_card

    return pred_dict

def subplan_to_sql(qrep, subplan_node):
    '''
    @ests: dict; key: label of the subplan. value: cardinality estimate.
    '''
    sg = qrep["join_graph"].subgraph(subplan_node)
    subsql = nx_graph_to_query(sg)
    return subsql
