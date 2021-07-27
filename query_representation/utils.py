import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
from moz_sql_parser import parse
import time
# from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
# from networkx.algorithms import bipartite
import networkx as nx
import itertools
import hashlib
import psycopg2 as pg
import shelve
import pdb
import os
import errno
import klepto
import getpass

# used for shortest-path or flow based framing of QO
# we add a new source node to the subset_graph, and add edges to each of the
# single table nodes; then a path from source to the node with all tables
# becomes a query plan etc.
SOURCE_NODE = tuple(["SOURCE"])

MAX_JOINS = 16
ALIAS_FORMAT = "{TABLE} AS {ALIAS}"
RANGE_PREDS = ["gt", "gte", "lt", "lte"]
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"

'''
functions copied over from ryan's utils files
'''

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
            yield "{} as {}".format(real_name, alias)
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
    count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
    return count_query

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
        if "!" in match:
            left, right = match.split("!=")
            if "." in right:
                # must be a join, so add it.
                join_clauses.append(left.strip() + " != " + right.strip())
            continue
        left, right = match.split("=")
        # ugh dumb hack
        if "." in right:
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
        print("moz sql parser failed to parse this!")
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

    token_list = sqlparse.sql.TokenList(wheres)

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
    except:
        print(explain)
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
    @timeout:
    @db_host: going to ignore it so default localhost is used.
    executes the given sql on the DB, and caches the results in a
    persistent store if it took longer than self.execution_cache_threshold.
    '''
    sql_cache = None
    if sql_cache_dir is not None:
        assert isinstance(sql_cache_dir, str)
        sql_cache = klepto.archives.dir_archive(sql_cache_dir,
                cached=True, serialized=True)

    hashed_sql = deterministic_hash(sql)

    # archive only considers the stuff stored in disk
    if sql_cache is not None and hashed_sql in sql_cache.archive:
        return sql_cache.archive[hashed_sql], False

    start = time.time()

    os_user = getpass.getuser()
    # con = pg.connect(user=user, port=port,
            # password=pwd, database=db_name)
    con = pg.connect(user=user, host=db_host, port=port,
            password=pwd, database=db_name)
    cursor = con.cursor()
    if timeout is not None:
        cursor.execute("SET statement_timeout = {}".format(timeout))
    try:
        cursor.execute(sql)
    except Exception as e:
        # print("query failed to execute: ", sql)
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
                # print(qrep["sql"])
                # print(node)
                # print(qrep["template_name"])
                # print(info["cardinality"])
                assert False
    return cards

