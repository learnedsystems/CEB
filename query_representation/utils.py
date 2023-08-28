import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
import time
import networkx as nx
import itertools
import hashlib
import psycopg2 as pg
import shelve
import pdb
import os
import errno
import getpass

import glob
from .query import *
import random


# used for shortest-path or flow based framing of QO
# we add a new source node to the subset_graph, and add edges to each of the
# single table nodes; then a path from source to the node with all tables
# becomes a query plan etc.
SOURCE_NODE = tuple(["SOURCE"])

MAX_JOINS = 16
ALIAS_FORMAT = "{TABLE} AS {ALIAS}"
RANGE_PREDS = ["gt", "gte", "lt", "lte"]
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"
REGEX_TEMPLATES = ['10a', '11a', '11b', '3b', '9b', '9a']
TIMEOUT_CARD = 150001000000

def update_job_parsing(qrep):
    '''
    fixes some error in the predicates parsed from JOB.
    '''
    for node,data in qrep["join_graph"].nodes(data=True):
        if "pred_vals" not in data:
            continue

        if len(data["predicates"]) != len(data["pred_vals"]):
            newvals = []
            newcols = []
            newtypes = []

            for di,dpred in enumerate(data["predicates"]):
                if "!=" in dpred:
                    newtypes.append("not eq")
                    dpreds = dpred.split("!=")
                    assert len(dpreds) == 2
                    newcols.append(dpreds[0])
                    newvals.append(dpreds[1])

            data["pred_vals"] += newvals
            data["pred_cols"] += newcols
            data["pred_types"] += newtypes

    for node,data in qrep["subset_graph"].nodes(data=True):
        if data["cardinality"]["actual"] == 0:
            data["cardinality"]["actual"] = 1

def load_rts():
    RTDIRS = ["/flash1/pari/MyCEB/runtime_plans/pg"]
    rtdfs = []
    for RTDIR in RTDIRS:
        rdirs = os.listdir(RTDIR)
        for rd in rdirs:
            rtfn = os.path.join(RTDIR, rd, "Runtimes.csv")
            if os.path.exists(rtfn):
                rtdfs.append(pd.read_csv(rtfn))

    rtdf = pd.concat(rtdfs)
    print("Num RTs: ", len(rtdf))
    return rtdf

def load_qdata_onlypg_plan(fns, data_params, skip_timeouts=False):
    qreps = []
    rtdf = load_rts()

    for qfn in fns:
        qrep = load_qrep(qfn)
        qname = os.path.basename(qrep["name"])
        if qname not in rtdf["qname"].values:
            continue
        tmp = rtdf[rtdf["qname"] == qrep["name"]]
        exp = tmp["exp_analyze"].values[0]
        try:
            exp = eval(exp)
        except:
            continue
        G = explain_to_nx(exp)
        seen_subplans = [ndata["aliases"] for n,ndata in
                G.nodes(data=True)]
        qrep["subplan_mask"] = seen_subplans

        if "job" in qfn and "joblight" not in qfn:
            update_job_parsing(qrep)

        skip = False
        for node in qrep["subset_graph"].nodes():
            if "cardinality" not in qrep["subset_graph"].nodes()[node]:
                skip = True
                break

            if "actual" not in qrep["subset_graph"].nodes()[node]["cardinality"]:
                skip = True
                continue

            if qrep["subset_graph"].nodes()[node]["cardinality"]["actual"] \
                    < 1:
                skip = True
                break

            if "expected" not in qrep["subset_graph"].nodes()[node]["cardinality"]:
                skip = True
                break

        if skip and skip_timeouts:
            continue

        qreps.append(qrep)
        template_name = os.path.basename(os.path.dirname(qfn))
        wkname = os.path.basename(os.path.dirname(os.path.dirname(qfn)))
        qrep["name"] = os.path.basename(qfn)
        qrep["template_name"] = template_name
        qrep["workload"] = wkname

    return qreps

def load_qdata(fns):
    qreps = []
    for qfn in fns:
        qrep = load_qrep(qfn)
        if "job" in qfn and "joblight" not in qfn:
            update_job_parsing(qrep)

        skip = False
        for node in qrep["subset_graph"].nodes():
            if "cardinality" not in qrep["subset_graph"].nodes()[node]:
                skip = True
                break
            if "actual" not in qrep["subset_graph"].nodes()[node]["cardinality"]:
                skip = True
                continue

            # if qrep["subset_graph"].nodes()[node]["cardinality"]["actual"] \
                    # >= TIMEOUT_CARD:
                # skip = True
                # continue

            # skips zeros
            if qrep["subset_graph"].nodes()[node]["cardinality"]["actual"] \
                    < 1:
                skip = True
                break

            if "expected" not in qrep["subset_graph"].nodes()[node]["cardinality"]:
                skip = True
                break

        if skip:
            continue

        qreps.append(qrep)
        template_name = os.path.basename(os.path.dirname(qfn))
        wkname = os.path.basename(os.path.dirname(os.path.dirname(qfn)))
        qrep["name"] = os.path.basename(qfn)
        qrep["template_name"] = template_name
        qrep["workload"] = wkname

    return qreps

from sklearn.model_selection import train_test_split
def get_query_splits(data_params):
    from types import SimpleNamespace
    data_params = SimpleNamespace(**data_params)

    fns = list(glob.glob(data_params.query_dir + "/*"))
    fns = [fn for fn in fns if os.path.isdir(fn)]
    skipped_templates = []
    train_qfns = []
    test_qfns = []
    val_qfns = []

    if data_params.no_regex_templates:
        new_templates = []
        for template_dir in fns:
            isregex = False
            for regtmp in REGEX_TEMPLATES:
                if regtmp in template_dir:
                    isregex = True
            if isregex:
                skipped_templates.append(template_dir)
            else:
                new_templates.append(template_dir)
        fns = new_templates

    if data_params.train_test_split_kind == "template":
        # the train/test split will be on the template names
        sorted_fns = copy.deepcopy(fns)
        sorted_fns.sort()
        train_tmps, test_tmps = train_test_split(sorted_fns,
                test_size=data_params.test_size,
                random_state=data_params.diff_templates_seed)

    elif data_params.train_test_split_kind == "custom":
        train_tmp_names = data_params.train_tmps.split(",")
        test_tmp_names = data_params.test_tmps.split(",")
        train_tmps = []
        test_tmps = []

        for fn in fns:
            for ctmp in train_tmp_names:
                if "/" + ctmp in fn:
                    train_tmps.append(fn)
                    break

            for ctmp in test_tmp_names:
                if "/" + ctmp in fn or ctmp == "all":
                    test_tmps.append(fn)
                    break

    for qi,qdir in enumerate(fns):
        if ".json" in qdir:
            continue
        if not os.path.isdir(qdir):
            continue

        template_name = os.path.basename(qdir)
        if data_params.query_templates != "all":
            query_templates = data_params.query_templates.split(",")
            if template_name not in query_templates:
                skipped_templates.append(template_name)
                continue
        if data_params.skip7a and template_name == "7a":
            skipped_templates.append(template_name)
            continue

        # let's first select all the qfns we are going to load
        qfns = list(glob.glob(qdir+"/*.pkl"))
        qfns.sort()

        if data_params.num_samples_per_template == -1 \
                or data_params.num_samples_per_template >= len(qfns):
            qfns = qfns
        elif data_params.num_samples_per_template < len(qfns):
            qfns = qfns[0:data_params.num_samples_per_template]
        else:
            assert False

        if data_params.train_test_split_kind == "template":
            cur_val_fns = []
            if qdir in train_tmps:
                cur_train_fns = qfns
                if data_params.val_size != 0.0:
                    cur_val_fns, cur_train_fns = train_test_split(cur_train_fns,
                            test_size=1-data_params.val_size,
                            random_state=data_params.seed)

                cur_test_fns = []
            elif qdir in test_tmps:
                cur_train_fns = []
                cur_test_fns = qfns
            else:
                continue
        elif data_params.train_test_split_kind == "custom":
            if qdir in train_tmps:
                cur_val_fns, cur_train_fns = train_test_split(qfns,
                        test_size=1-data_params.val_size,
                        random_state=data_params.seed)
                cur_test_fns = []
            elif qdir in test_tmps:
                # no validation set from here
                cur_val_fns = []
                cur_train_fns = []
                cur_test_fns = qfns
            else:
                continue

        elif data_params.train_test_split_kind == "query":
            if data_params.val_size == 0.0:
                cur_val_fns = []
            else:
                cur_val_fns, qfns = train_test_split(qfns,
                        test_size=1-data_params.val_size,
                        random_state=data_params.diff_templates_seed)

            if data_params.test_size == 0:
                cur_test_fns = []
                cur_train_fns = qfns
            else:
                cur_train_fns, cur_test_fns = train_test_split(qfns,
                        test_size=data_params.test_size,
                        random_state=data_params.diff_templates_seed)

        train_qfns += cur_train_fns
        val_qfns += cur_val_fns
        test_qfns += cur_test_fns

    print("Skipped templates: ", " ".join(skipped_templates))
    trainqnames = [os.path.basename(qfn) for qfn in train_qfns]

    eval_qfns = []
    eval_qdirs = data_params.eval_query_dir.split(",")
    for qdir in eval_qdirs:
        if qdir == "":
            eval_qfns.append([])
            continue

        if "imdb" in qdir and not \
            ("imdb-unique-plans1950" in data_params.query_dir or \
                    "imdb-unique-plans1980" in data_params.query_dir or \
                    "1a" in data_params.query_templates):
            # with open("ceb_runtime_qnames.pkl", "rb") as f:
                # qkeys = pickle.load(f)
                # with open('ceb_runtimes_qnames.txt', 'w') as f:
                    # for line in qkeys:
                        # f.writeline(f"{line}\n")
            with open(os.path.join("queries", "ceb_runtime_qnames.txt"), "r") as f:
                qkeys = f.read()
            qkeys = qkeys.split("\n")
            print("going to read only {} CEB queries".format(len(qkeys)-1))

        elif "ergast" in qdir:
            with open("ergast_runtime_qnames.pkl", "rb") as f:
                qkeys = pickle.load(f)
        else:
            qkeys = None

        cur_eval_qfns = []
        fns = list(glob.glob(qdir + "/*"))
        fns = [fn for fn in fns if os.path.isdir(fn)]

        for qi,qdir in enumerate(fns):
            if ".json" in qdir:
                continue

            template_name = os.path.basename(qdir)
            if data_params.eval_templates != "all" and \
                template_name not in data_params.eval_templates.split(","):
                print("skipping eval template: ", template_name)
                continue

            if data_params.skip7a and template_name == "7a":
                skipped_templates.append(template_name)
                continue

            # let's first select all the qfns we are going to load
            qfns = list(glob.glob(qdir+"/*.pkl"))
            qfns.sort()

            if qkeys is not None:
                qfns = [qf for qf in qfns if os.path.basename(qf) in qkeys]

            if ("imdb-unique-plans1950" in data_params.query_dir or \
                    "imdb-unique-plans1980" in data_params.query_dir or \
                    "1a" in data_params.query_templates):
                qfns = [qf for qf in qfns if os.path.basename(qf) not in trainqnames]

            cur_eval_qfns += qfns

        random.shuffle(cur_eval_qfns)
        eval_qfns.append(cur_eval_qfns)

    if data_params.train_test_split_kind == "query":
        pass
    else:
        train_tmp_names = [os.path.basename(tfn) for tfn in train_tmps]
        test_tmp_names = [os.path.basename(tfn) for tfn in test_tmps]

        print("""Selected {} train templates, {} test templates"""\
                .format(len(train_tmp_names), len(test_tmp_names)))
        print("""Training templates: {}\nEvaluation templates: {}""".\
                format(",".join(train_tmp_names), ",".join(test_tmp_names)))

    # going to shuffle all these lists, so queries are evenly distributed. Plan
    # Cost functions for some of these templates take a lot longer; so when we
    # compute them in parallel, we want the queries to be shuffled so the
    # workload is divided evely
    random.shuffle(train_qfns)
    random.shuffle(test_qfns)
    random.shuffle(val_qfns)

    return train_qfns, test_qfns, val_qfns, eval_qfns

def _find_all_tables(plan):
    '''
    '''
    # find all the scan nodes under the current level, and return those
    table_names = extract_values(plan, "Relation Name")
    alias_names = extract_values(plan, "Alias")
    table_names.sort()
    alias_names.sort()

    return table_names, alias_names

def extract_aliases2(plan):
    aliases = extract_values(plan, "Alias")
    return aliases

def explain_to_nx(explain):
    '''
    '''
    # JOIN_KEYS = ["Hash Join", "Nested Loop", "Join"]
    base_table_nodes = []
    join_nodes = []

    def _get_node_name(tables):
        name = ""
        if len(tables) > 1:
            name = str(deterministic_hash(str(tables)))[0:5]
            join_nodes.append(name)
        else:
            name = tables[0]
            if len(name) >= 6:
                # no aliases, shorten it
                name = "".join([n[0] for n in name.split("_")])
                if name in base_table_nodes:
                    name = name + "2"
            base_table_nodes.append(name)
        return name

    def _add_node_stats(node, plan):
        # add stats for the join
        G.nodes[node]["Plan Rows"] = plan["Plan Rows"]
        if "Actual Rows" in plan:
            G.nodes[node]["Actual Rows"] = plan["Actual Rows"]
        else:
            G.nodes[node]["Actual Rows"] = -1.0
        if "Actual Total Time" in plan:
            G.nodes[node]["total_time"] = plan["Actual Total Time"]

            if "Plans" not in plan:
                children_time = 0.0
            elif len(plan["Plans"]) == 2:
                children_time = plan["Plans"][0]["Actual Total Time"] \
                        + plan["Plans"][1]["Actual Total Time"]
            elif len(plan["Plans"]) == 1:
                children_time = plan["Plans"][0]["Actual Total Time"]
            else:
                assert False

            G.nodes[node]["cur_time"] = plan["Actual Total Time"]-children_time

        else:
            G.nodes[node]["Actual Total Time"] = -1.0

        if "Node Type" in plan:
            G.nodes[node]["Node Type"] = plan["Node Type"]

        total_cost = plan["Total Cost"]
        G.nodes[node]["Total Cost"] = total_cost
        aliases = G.nodes[node]["aliases"]
        if len(G.nodes[node]["tables"]) > 1:
            children_cost = plan["Plans"][0]["Total Cost"] \
                    + plan["Plans"][1]["Total Cost"]

            # +1 to avoid cases which are very close
            if not total_cost+1 >= children_cost:
                print("aliases: {} children cost: {}, total cost: {}".format(\
                        aliases, children_cost, total_cost))
                # pdb.set_trace()
            G.nodes[node]["cur_cost"] = total_cost - children_cost
            G.nodes[node]["node_label"] = plan["Node Type"][0]
            G.nodes[node]["scan_type"] = ""
        else:
            G.nodes[node]["cur_cost"] = total_cost
            G.nodes[node]["node_label"] = node
            # what type of scan was this?
            node_types = extract_values(plan, "Node Type")
            for i, full_n in enumerate(node_types):
                shortn = ""
                for n in full_n.split(" "):
                    shortn += n[0]
                node_types[i] = shortn

            scan_type = "\n".join(node_types)
            G.nodes[node]["scan_type"] = scan_type

    def traverse(obj):
        if isinstance(obj, dict):
            if "Plans" in obj:
                if len(obj["Plans"]) == 2:
                    # these are all the joins
                    left_tables, left_aliases = _find_all_tables(obj["Plans"][0])
                    right_tables, right_aliases = _find_all_tables(obj["Plans"][1])
                    if len(left_tables) == 0 or len(right_tables) == 0:
                        return
                    all_tables = left_tables + right_tables
                    all_aliases = left_aliases + right_aliases
                    all_aliases.sort()
                    all_tables.sort()

                    if len(left_aliases) > 0:
                        node0 = _get_node_name(left_aliases)
                        node1 = _get_node_name(right_aliases)
                        node_new = _get_node_name(all_aliases)
                    else:
                        node0 = _get_node_name(left_tables)
                        node1 = _get_node_name(right_tables)
                        node_new = _get_node_name(all_tables)

                    # update graph
                    # G.add_edge(node0, node_new)
                    # G.add_edge(node1, node_new)
                    G.add_edge(node_new, node0)
                    G.add_edge(node_new, node1)
                    G.edges[(node_new, node0)]["join_direction"] = "left"
                    G.edges[(node_new, node1)]["join_direction"] = "right"

                    # add other parameters on the nodes
                    G.nodes[node0]["tables"] = left_tables
                    G.nodes[node1]["tables"] = right_tables
                    G.nodes[node0]["aliases"] = left_aliases
                    G.nodes[node1]["aliases"] = right_aliases
                    G.nodes[node_new]["tables"] = all_tables
                    G.nodes[node_new]["aliases"] = all_aliases

                    # TODO: if either the left, or right were a scan, then add
                    # scan stats
                    _add_node_stats(node_new, obj)

                    if len(left_tables) == 1:
                        _add_node_stats(node0, obj["Plans"][0])
                    if len(right_tables) == 1:
                        _add_node_stats(node1, obj["Plans"][1])

            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    traverse(v)

        elif isinstance(obj, list) or isinstance(obj,tuple):
            for item in obj:
                traverse(item)

    G = nx.DiGraph()
    traverse(explain)
    G.base_table_nodes = base_table_nodes
    G.join_nodes = join_nodes
    return G

def connected_subgraphs(g):
    # for i in range(2, len(g)+1):
    for i in range(1, len(g)+1):
        for nodes_in_sg in itertools.combinations(g.nodes, i):
            sg = g.subgraph(nodes_in_sg)
            if nx.is_connected(sg):
                yield tuple(sorted(sg.nodes))

def generate_subset_graph(g):
    subset_graph = nx.DiGraph()
    for csg in connected_subgraphs(g):
        subset_graph.add_node(csg)
    # group by size
    max_subgraph_size = max(len(x) for x in subset_graph.nodes)
    subgraph_groups = [[] for _ in range(max_subgraph_size)]
    for node in subset_graph.nodes:
        subgraph_groups[len(node)-1].append(node)

    for g1, g2 in zip(subgraph_groups, subgraph_groups[1:]):
        for superset in g2:
            super_as_set = set(superset)
            for subset in g1:
                assert len(superset) == len(subset) + 1
                if set(subset) < super_as_set:
                    subset_graph.add_edge(superset, subset)

    return subset_graph

def get_optimal_edges(sg):
    paths = {}
    orig_sg = sg
    sg = sg.copy()
    while len(sg.nodes) != 0:
        # first, find the root(s) of the subgraph at the highest level
        roots = {n for n,d in sg.in_degree() if d == 0}
        max_size_root = len(max(roots, key=lambda x: len(x)))
        roots = {r for r in roots if len(r) == max_size_root}

        # find everything within reach of 1
        reach_1 = set()
        for root in roots:
            reach_1.update(sg.neighbors(root))

        # build a bipartite graph and do the matching
        all_nodes = reach_1 | roots
        bipart_layer = sg.subgraph(all_nodes).to_undirected()
        assert(bipartite.is_bipartite(bipart_layer))
        matching = bipartite.hopcroft_karp_matching(bipart_layer, roots)
        matching = { k: v for k,v in matching.items() if k in roots}

        # sanity check -- every vertex should appear in exactly one path
        assert len(set(matching.values())) == len(matching)

        # find unmatched roots and add a path to $, indicating that
        # the path has terminated.
        for unmatched_root in roots - matching.keys():
            matching[unmatched_root] = "$"
        assert len(matching) == len(roots)

        # sanity check -- nothing was already in our paths
        for k, v in matching.items():
            assert k not in paths.keys()
            assert v not in paths.keys()
            assert v == "$" or v not in paths.values()

        # sanity check -- all roots have an edge assigned
        for root in roots:
            assert root in matching.keys()

        paths.update(matching)

        # remove the old roots
        sg.remove_nodes_from(roots)
    return paths

def reconstruct_paths(edges):
    g = nx.Graph()
    for pair in edges.items():
        g.add_nodes_from(pair)

    for v1, v2 in edges.items():
        if v2 != "$":
            assert len(v1) > len(v2) and set(v1) > set(v2)
        g.add_edge(v1, v2)


    if "$" in g.nodes:
        g.remove_node("$")

    # for node in g.nodes:
        # assert g.degree(node) <= 2, f"{node} had degree of {g.degree(node)}"

    conn_comp = nx.algorithms.components.connected_components(g)
    paths = (sorted(x, key=len, reverse=True) for x in conn_comp)
    return paths

def greedy(subset_graph, plot=False):
    subset_graph = subset_graph.copy()

    while subset_graph:
        longest_path = nx.algorithms.dag.dag_longest_path(subset_graph)
        if plot:
            display(draw_graph(subset_graph, highlight_nodes=longest_path))
        subset_graph.remove_nodes_from(longest_path)
        yield longest_path

def path_to_join_order(path):
    remaining = set(path[0])
    for node in path[1:]:
        diff = remaining - set(node)
        yield diff
        remaining -= diff
    yield remaining

def order_to_from_clause(join_graph, join_order, alias_mapping):
    clauses = []
    for rels in join_order:
        if len(rels) > 1:
            # we should ask PG for an ordering here, since there's
            # no way to specify that the optimizer should control only these
            # bottom-level joins.
            sg = join_graph.subgraph(rels)
            sql = nx_graph_to_query(sg)
            con = pg.connect(user="ubuntu", host="localhost", database="imdb")
            cursor = con.cursor()
            # cursor.execute(f"explain (format json) {sql}")
            cursor.execute("explain (format json) {}".format(sql))
            explain = cursor.fetchall()
            cursor.close()
            con.close()
            pg_order,_,_ = get_pg_join_order(join_graph, explain)
            assert not clauses
            clauses.append(pg_order)
            continue

        # clause = f"{alias_mapping[rels[0]]} as {rels[0]}"
        clause = "{} as {}".format(alias_mapping[rels[0]], rels[0])
        clauses.append(clause)

    return " CROSS JOIN ".join(clauses)

join_types = set(["Nested Loop", "Hash Join", "Merge Join", "Index Scan",\
        "Seq Scan", "Bitmap Heap Scan"])

def extract_aliases(plan, jg=None):
    if "Alias" in plan:
        assert plan["Node Type"] == "Bitmap Heap Scan" or "Plans" not in plan
        if jg:
            alias = plan["Alias"]
            real_name = jg.nodes[alias]["real_name"]
            # yield f"{real_name} as {alias}"
            # yield "{} as {}".format(real_name, alias)
            yield "\"{}\" as {}".format(real_name, alias)
        else:
            yield plan["Alias"]

    if "Plans" not in plan:
        return

    for subplan in plan["Plans"]:
        yield from extract_aliases(subplan, jg=jg)

def analyze_plan(plan):
    if plan["Node Type"] in join_types:
        aliases = extract_aliases(plan)
        data = {"aliases": list(sorted(aliases))}
        if "Plan Rows" in plan:
            data["expected"] = plan["Plan Rows"]
        if "Actual Rows" in plan:
            data["actual"] = plan["Actual Rows"]
        else:
            print("Actual Rows not in plan!")
            pdb.set_trace()

        yield data

    if "Plans" not in plan:
        return

    for subplan in plan["Plans"]:
        yield from analyze_plan(subplan)

'''
functions copied over from pari's util files
'''

def nodes_to_sql(nodes, join_graph):
    alias_mapping = {}
    for node_set in nodes:
        for node in node_set:
            alias_mapping[node] = join_graph.nodes[node]["real_name"]

    from_clause = order_to_from_clause(join_graph, nodes, alias_mapping)

    subg = join_graph.subgraph(alias_mapping.keys())
    assert nx.is_connected(subg)

    sql_str = nx_graph_to_query(subg, from_clause=from_clause)
    return sql_str

def nx_graph_to_query(G, from_clause=None):
    '''
    @G: join_graph in the query_represntation format
    '''

    froms = []
    conds = []
    for nd in G.nodes(data=True):
        node = nd[0]
        data = nd[1]
        if "real_name" in data:
            froms.append(ALIAS_FORMAT.format(TABLE=data["real_name"],
                                             ALIAS=node))
        else:
            froms.append(node)

        for pred in data["predicates"]:
            if pred not in conds:
                conds.append(pred)

    for edge in G.edges(data=True):
        conds.append(edge[2]['join_condition'])

    # preserve order for caching
    froms.sort()
    conds.sort()
    from_clause = " , ".join(froms) if from_clause is None else from_clause
    if len(conds) > 0:
        wheres = ' AND '.join(conds)
        from_clause += " WHERE " + wheres

    if "aggr_cmd" not in G.graph or G.graph["aggr_cmd"] == "":
        ret_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
    else:
        SQL_TMP = "{} FROM {}"
        ret_query = SQL_TMP.format(G.graph["aggr_cmd"],
                        from_clause)

    return ret_query

# def extract_join_clause(query):
    # '''
    # FIXME: this can be optimized further / or made to handle more cases
    # '''
    # parsed = sqlparse.parse(query)[0]
    # # let us go over all the where clauses
    # start = time.time()
    # where_clauses = None
    # for token in parsed.tokens:
        # if (type(token) == sqlparse.sql.Where):
            # where_clauses = token
    # if where_clauses is None:
        # return []
    # join_clauses = []

    # froms, aliases, table_names = extract_from_clause(query)
    # if len(aliases) > 0:
        # tables = [k for k in aliases]
    # else:
        # tables = table_names
    # matches = find_all_clauses(tables, where_clauses)

    # for match in matches:
        # if "=" not in match or match.count("=") > 1:
            # continue
        # if "<=" in match or ">=" in match:
            # continue
        # match = match.replace(";", "")
        # if "!" in match:
            # left, right = match.split("!=")
            # if "." in right:
                # # must be a join, so add it.
                # join_clauses.append(left.strip() + " != " + right.strip())
            # continue
        # left, right = match.split("=")

        # # ugh dumb hack
        # if "." in right:
            # # must be a join, so add it.
            # join_clauses.append(left.strip() + " = " + right.strip())

    # return join_clauses

def extract_join_clause(query):
    '''
    FIXME: this can be optimized further / or made to handle more cases
    '''
    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    start = time.time()
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    if where_clauses is None:
        return []
    join_clauses = []

    froms, aliases, table_names = extract_from_clause(query)
    if len(aliases) > 0:
        tables = [k for k in aliases]
    else:
        tables = table_names
    matches = find_all_clauses(tables, where_clauses)

    for match in matches:
        if "=" not in match or match.count("=") > 1:
            continue
        if "<=" in match or ">=" in match:
            continue
        match = match.replace(";", "")

        if "!=" in match:
            left, right = match.split("!=")

            if not ("id" in left.lower() and "id" in right.lower()):
                continue

            if right.count(".") == 1 and "'" not in right:
                # must be a join, so add it.
                join_clauses.append(left.strip() + " != " + right.strip())
            continue

        left, right = match.split("=")

        if not ("id" in left.lower() and "id" in right.lower()):
            continue

        # ugh dumb hack
        if right.count(".") == 1 and "'" not in right:
            # must be a join, so add it.
            join_clauses.append(left.strip() + " = " + right.strip())

    return join_clauses

def get_all_wheres(parsed_query):
    pred_vals = []
    if "where" not in parsed_query:
        pass
    elif "and" not in parsed_query["where"]:
        pred_vals = [parsed_query["where"]]
    else:
        pred_vals = parsed_query["where"]["and"]
    return pred_vals

def extract_predicates(query):
    '''
    @ret:
        - column names with predicate conditions in WHERE.
        - predicate operator type (e.g., "in", "lte" etc.)
        - predicate value
    Note: join conditions don't count as predicate conditions.

    FIXME: temporary hack. For range queries, always returning key
    "lt", and vals for both the lower and upper bound
    '''
    def parse_column(pred, cur_pred_type):
        '''
        gets the name of the column, and whether column location is on the left
        (0) or right (1)
        '''
        for i, obj in enumerate(pred[cur_pred_type]):
            assert i <= 1
            if isinstance(obj, str) and "." in obj:
                # assert "." in obj
                column = obj
            elif isinstance(obj, dict):
                assert "literal" in obj
                val = obj["literal"]
                val_loc = i
            else:
                val = obj
                val_loc = i

        assert column is not None
        assert val is not None
        return column, val_loc, val

    def _parse_predicate(pred, pred_type):
        if pred_type == "eq":
            columns = pred[pred_type]
            if len(columns) <= 1:
                return None
            # FIXME: more robust handling?
            if "." in str(columns[1]):
                # should be a join, skip this.
                # Note: joins only happen in "eq" predicates
                return None
            predicate_types.append(pred_type)
            predicate_cols.append(columns[0])
            predicate_vals.append(columns[1])

        elif pred_type in RANGE_PREDS:
            vals = [None, None]
            col_name, val_loc, val = parse_column(pred, pred_type)
            vals[val_loc] = val

            # this loop may find no matching predicate for the other side, in
            # which case, we just leave the val as None
            for pred2 in pred_vals:
                pred2_type = list(pred2.keys())[0]
                if pred2_type in RANGE_PREDS:
                    col_name2, val_loc2, val2 = parse_column(pred2, pred2_type)
                    if col_name2 == col_name:
                        # assert val_loc2 != val_loc
                        if val_loc2 == val_loc:
                            # same predicate as pred
                            continue
                        vals[val_loc2] = val2
                        break

            predicate_types.append("lt")
            predicate_cols.append(col_name)
            if "g" in pred_type:
                # reverse vals, since left hand side now means upper bound
                vals.reverse()
            predicate_vals.append(vals)

        elif pred_type == "between":
            # we just treat it as a range query
            col = pred[pred_type][0]
            val1 = pred[pred_type][1]
            val2 = pred[pred_type][2]
            vals = [val1, val2]
            predicate_types.append("lt")
            predicate_cols.append(col)
            predicate_vals.append(vals)
        elif pred_type == "in" \
                or "like" in pred_type:
            # includes preds like, ilike, nlike etc.
            column = pred[pred_type][0]
            # what if column has been seen before? Will just be added again to
            # the list of predicates, which is the correct behaviour
            vals = pred[pred_type][1]
            if isinstance(vals, dict):
                vals = vals["literal"]
            if not isinstance(vals, list):
                vals = [vals]
            predicate_types.append(pred_type)
            predicate_cols.append(column)
            predicate_vals.append(vals)
        elif pred_type == "or":
            for pred2 in pred[pred_type]:
                # print(pred2)
                assert len(pred2.keys()) == 1
                pred_type2 = list(pred2.keys())[0]
                _parse_predicate(pred2, pred_type2)

        elif pred_type == "missing":
            column = pred[pred_type]
            val = ["NULL"]
            predicate_types.append("in")
            predicate_cols.append(column)
            predicate_vals.append(val)
        else:
            # assert False
            # TODO: need to support "OR" statements
            return None
            # assert False, "unsupported predicate type"

    start = time.time()
    predicate_cols = []
    predicate_types = []
    predicate_vals = []
    if "::float" in query:
        query = query.replace("::float", "")
    elif "::int" in query:
        query = query.replace("::int", "")
    # really fucking dumb
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in query:
        query = query.replace(bad_str1, "")

    if bad_str2 in query:
        query = query.replace(bad_str2, "")

    try:
        parsed_query = parse(query)
    except:
        print(query)
        print("sql parser failed to parse this!")
        pdb.set_trace()
    pred_vals = get_all_wheres(parsed_query)

    for i, pred in enumerate(pred_vals):
        try:
            assert len(pred.keys()) == 1
        except:
            print(pred)
            pdb.set_trace()
        pred_type = list(pred.keys())[0]
        # if pred == "or" or pred == "OR":
            # continue
        _parse_predicate(pred, pred_type)

    return predicate_cols, predicate_types, predicate_vals

def extract_from_clause(query):
    '''
    Optimized version using sqlparse.
    Extracts the from statement, and the relevant joins when there are multiple
    tables.
    @ret: froms:
          froms: [alias1, alias2, ...] OR [table1, table2,...]
          aliases:{alias1: table1, alias2: table2} (OR [] if no aliases present)
          tables: [table1, table2, ...]
    '''
    def handle_table(identifier):
        table_name = identifier.get_real_name()
        alias = identifier.get_alias()
        tables.append(table_name)
        if alias is not None:
            from_clause = ALIAS_FORMAT.format(TABLE = table_name,
                                ALIAS = alias)
            froms.append(from_clause)
            aliases[alias] = table_name
        else:
            froms.append(table_name)

    start = time.time()
    froms = []
    # key: alias, val: table name
    aliases = {}
    # just table names
    tables = []

    start = time.time()
    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    from_token = None
    from_seen = False
    for token in parsed.tokens:
        if from_seen:
            if isinstance(token, IdentifierList) or isinstance(token,
                    Identifier):
                from_token = token
                break
        if token.ttype is Keyword and token.value.upper() == 'FROM':
            from_seen = True
    assert from_token is not None
    if isinstance(from_token, IdentifierList):
        for identifier in from_token.get_identifiers():
            handle_table(identifier)
    elif isinstance(from_token, Identifier):
        handle_table(from_token)
    else:
        assert False

    return froms, aliases, tables

def find_next_match(tables, wheres, index):
    '''
    ignore everything till next
    '''
    match = ""
    _, token = wheres.token_next(index)
    if token is None:
        return None, None
    # FIXME: is this right?
    if token.is_keyword:
        index, token = wheres.token_next(index)

    tables_in_pred = find_all_tables_till_keyword(token)
    assert len(tables_in_pred) <= 2

    token_list = sqlparse.sql.TokenList(wheres).tokens

    while True:
        index, token = token_list.token_next(index)
        if token is None:
            break
        # print("token.value: ", token.value)
        if token.value.upper() == "AND":
            break

        match += " " + token.value

        if (token.value.upper() == "BETWEEN"):
            # ugh ugliness
            index, a = token_list.token_next(index)
            index, AND = token_list.token_next(index)
            index, b = token_list.token_next(index)
            match += " " + a.value
            match += " " + AND.value
            match += " " + b.value
            # Note: important not to break here! Will break when we hit the
            # "AND" in the next iteration.

    # print("tables: ", tables)
    # print("match: ", match)
    # print("tables in pred: ", tables_in_pred)
    for table in tables_in_pred:
        if table not in tables:
            # print(tables)
            # print(table)
            # pdb.set_trace()
            # print("returning index, None")
            return index, None

    if len(tables_in_pred) == 0:
        return index, None

    return index, match

def find_all_clauses(tables, wheres):
    matched = []
    # print(tables)
    index = 0
    while True:
        index, match = find_next_match(tables, wheres, index)
        # print("got index, match: ", index)
        # print(match)
        if match is not None:
            matched.append(match)
        if index is None:
            break

    return matched

def find_all_tables_till_keyword(token):
    tables = []
    # print("fattk: ", token)
    index = 0
    while (True):
        if (type(token) == sqlparse.sql.Comparison):
            left = token.left
            right = token.right
            if (type(left) == sqlparse.sql.Identifier):
                tables.append(left.get_parent_name())
            if (type(right) == sqlparse.sql.Identifier):
                tables.append(right.get_parent_name())
            break
        elif (type(token) == sqlparse.sql.Identifier):
            tables.append(token.get_parent_name())
            break
        try:
            index, token = token.token_next(index)
            if ("Literal" in str(token.ttype)) or token.is_keyword:
                break
        except:
            break

    return tables

def execute_query(sql, user, db_host, port, pwd, db_name, pre_execs):
    '''
    @db_host: going to ignore it so default localhost is used.
    @pre_execs: options like set join_collapse_limit to 1 that are executed
    before the query.
    '''
    con = pg.connect(user=user, host=db_host, port=port,
            password=pwd, database=db_name)
    cursor = con.cursor()

    for setup_sql in pre_execs:
        cursor.execute(setup_sql)

    try:
        cursor.execute(sql)
    except Exception as e:
        print(e)
        try:
            # con.commit()
            cursor.close()
            con.close()
        finally:
            if not "timeout" in str(e):
                print("failed to execute for reason other than timeout")
                print(e)
                return e
            return "timeout"

    exp_output = cursor.fetchall()
    cursor.close()
    con.close()

    return exp_output

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_pg_join_order(join_graph, explain):
    '''
    '''
    physical_join_ops = {}
    scan_ops = {}
    def __update_scan(plan):
        node_types = extract_values(plan, "Node Type")
        alias = extract_values(plan, "Alias")[0]
        for nt in node_types:
            if "Scan" in nt:
                scan_type = nt
                break
        scan_ops[alias] = nt

    def __extract_jo(plan):
        if plan["Node Type"] in join_types:
            left = list(extract_aliases(plan["Plans"][0], jg=join_graph))
            right = list(extract_aliases(plan["Plans"][1], jg=join_graph))
            all_froms = left + right
            all_nodes = []
            for from_clause in all_froms:
                from_alias = from_clause[from_clause.find(" as ")+4:]
                if "_info" in from_alias:
                    print(from_alias)
                    pdb.set_trace()
                all_nodes.append(from_alias)
            all_nodes.sort()
            all_nodes = " ".join(all_nodes)
            physical_join_ops[all_nodes] = plan["Node Type"]

            if len(left) == 1 and len(right) == 1:
                __update_scan(plan["Plans"][0])
                __update_scan(plan["Plans"][1])
                return left[0] +  " CROSS JOIN " + right[0]

            if len(left) == 1:
                __update_scan(plan["Plans"][0])
                return left[0] + " CROSS JOIN (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                __update_scan(plan["Plans"][1])
                return "(" + __extract_jo(plan["Plans"][0]) + ") CROSS JOIN " + right[0]

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") CROSS JOIN ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    try:
        return __extract_jo(explain[0][0][0]["Plan"]), physical_join_ops, scan_ops
    except Exception as e:
        # print(explain)
        print(e)
        pdb.set_trace()

def extract_join_graph(sql):
    '''
    @sql: string
    '''
    froms,aliases,tables = extract_from_clause(sql)
    joins = extract_join_clause(sql)
    join_graph = nx.Graph()

    for j in joins:
        j1 = j.split("=")[0]
        j2 = j.split("=")[1]
        t1 = j1[0:j1.find(".")].strip()
        t2 = j2[0:j2.find(".")].strip()
        try:
            assert t1 in tables or t1 in aliases
            assert t2 in tables or t2 in aliases
        except:
            print(t1, t2)
            print(tables)
            print(joins)
            print("table not in tables!")
            pdb.set_trace()

        join_graph.add_edge(t1, t2)
        join_graph[t1][t2]["join_condition"] = j

        if t1 in aliases:
            table1 = aliases[t1]
            table2 = aliases[t2]

            join_graph.nodes()[t1]["real_name"] = table1
            join_graph.nodes()[t2]["real_name"] = table2

    parsed = sqlparse.parse(sql)[0]
    # let us go over all the where clauses
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    assert where_clauses is not None

    if len(join_graph.nodes()) == 0:
        for alias in aliases:
            join_graph.add_node(alias)
            join_graph.nodes()[alias]["real_name"] = aliases[alias]

    for t1 in join_graph.nodes():
        tables = [t1]
        matches = find_all_clauses(tables, where_clauses)
        join_graph.nodes()[t1]["predicates"] = matches

    return join_graph

def extract_values(obj, key):
    """Recursively pull values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Return all matching values in an object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    # if "Scan" in v:
                        # print(v)
                        # pdb.set_trace()
                    # if "Join" in v:
                        # print(obj)
                        # pdb.set_trace()
                    arr.append(v)

        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

def cached_execute_query(sql, user, db_host, port, pwd, db_name,
        execution_cache_threshold, sql_cache_dir=None,
        timeout=120000):
    '''
    Note: removed the cache to get rid of klepto dependency.
    @timeout:
    @db_host: going to ignore it so default localhost is used.
    executes the given sql on the DB, and caches the results in a
    persistent store if it took longer than self.execution_cache_threshold.
    '''
    hashed_sql = deterministic_hash(sql)
    start = time.time()

    os_user = getpass.getuser()
    con = pg.connect(user=user, host=db_host, port=port,
            password=pwd, database=db_name)
    cursor = con.cursor()
    if timeout is not None:
        cursor.execute("SET statement_timeout = {}".format(timeout))
    try:
        cursor.execute(sql)
    except Exception as e:
        print(e)
        print("query failed to execute: ", sql)
        # FIXME: better way to do this.
        cursor.execute("ROLLBACK")
        con.commit()
        cursor.close()
        con.close()
        return None

    exp_output = cursor.fetchall()
    cursor.close()
    con.close()
    end = time.time()
    if (end - start > execution_cache_threshold) \
            and sql_cache is not None:
        sql_cache.archive[hashed_sql] = exp_output
    return exp_output

def extract_values(obj, key):
    """Recursively pull values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Return all matching values in an object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    # if "Scan" in v:
                        # print(v)
                        # pdb.set_trace()
                    # if "Join" in v:
                        # print(obj)
                        # pdb.set_trace()
                    arr.append(v)

        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

def is_float(val):
    try:
        float(val)
        return True
    except:
        return False

def extract_ints_from_string(string):
    return re.findall(r'\d+', string)

def get_all_cardinalities(samples, ckey):
    cards = []
    for qrep in samples:
        for node, info in qrep["subset_graph"].nodes().items():
            if node == SOURCE_NODE:
                continue
            cards.append(info[ckey]["actual"])
            if cards[-1] == 0:
                assert False
    return cards
