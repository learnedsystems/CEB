import pdb

NILJ_CONSTANT = 0.001
MAX_JOINS = 20

def set_cost_model(cursor, cost_model):
    '''
    This is for setting up PostgreSQL for evaluating the Postgres Plan Cost,
    and the runtimes.
    '''
    # makes things easier to understand
    cursor.execute("SET geqo_threshold = {}".format(MAX_JOINS))
    cursor.execute("SET join_collapse_limit = {}".format(MAX_JOINS))
    cursor.execute("SET from_collapse_limit = {}".format(MAX_JOINS))

    if cost_model == "cm1" or cost_model == "C":
        cursor.execute("SET max_parallel_workers = 0")
        cursor.execute("SET max_parallel_workers_per_gather = 0")

        cursor.execute("SET enable_material = off")
        cursor.execute("SET enable_hashjoin = on")
        cursor.execute("SET enable_mergejoin = on")
        cursor.execute("SET enable_nestloop = on")

        cursor.execute("SET enable_indexscan = {}".format("on"))
        cursor.execute("SET enable_seqscan = {}".format("on"))
        cursor.execute("SET enable_indexonlyscan = {}".format("on"))
        cursor.execute("SET enable_bitmapscan = {}".format("on"))
        cursor.execute("SET enable_tidscan = {}".format("on"))

    elif cost_model == "C2":
        cursor.execute("SET max_parallel_workers = 0")
        cursor.execute("SET max_parallel_workers_per_gather = 0")

        cursor.execute("SET enable_material = off")
        cursor.execute("SET enable_hashjoin = on")
        cursor.execute("SET enable_mergejoin = on")
        cursor.execute("SET enable_nestloop = on")

        cursor.execute("SET enable_indexscan = {}".format("off"))
        cursor.execute("SET enable_seqscan = {}".format("on"))
        cursor.execute("SET enable_indexonlyscan = {}".format("off"))
        cursor.execute("SET enable_bitmapscan = {}".format("off"))
        cursor.execute("SET enable_tidscan = {}".format("off"))
    else:
        assert False, "{} cost model unknown".format(cost_model)

def add_single_node_edges(subset_graph, source):
    subset_graph.add_node(source)
    subset_graph.nodes()[source]["cardinality"] = {}
    subset_graph.nodes()[source]["cardinality"]["actual"] = 1.0

    for node in subset_graph.nodes():
        if len(node) != 1:
            continue
        if node[0] == source[0]:
            continue
        subset_graph.add_edge(node, source, cost=0.0)
        in_edges = subset_graph.in_edges(node)
        out_edges = subset_graph.out_edges(node)

        # if we need to add edges between single table nodes and rest
        for node2 in subset_graph.nodes():
            if len(node2) != 2:
                continue
            if node[0] in node2:
                subset_graph.add_edge(node2, node)

def update_subplan_costs(subset_graph, cost_model,
        cost_key="cost", ests=None):
    '''
    @computes costs based on a simple cost model.
    '''
    total_cost = 0.0
    cost_key = cost_model + cost_key
    for edge in subset_graph.edges():
        if len(edge[0]) == len(edge[1]):
            # special case: source node --> single table node edges
            subset_graph[edge[0]][edge[1]][cost_key] = 1.0
            continue

        if len(edge[1]) > len(edge[0]):
            print(edge)
            pdb.set_trace()

        assert len(edge[1]) < len(edge[0])

        node1 = edge[1]
        diff = set(edge[0]) - set(edge[1])
        node2 = list(diff)
        node2.sort()
        node2 = tuple(node2)
        assert node2 in subset_graph.nodes()
        # joined node
        node3 = edge[0]
        cards1 = subset_graph.nodes()[node1]["cardinality"]
        cards2 = subset_graph.nodes()[node2]["cardinality"]
        cards3 = subset_graph.nodes()[edge[0]]["cardinality"]

        if isinstance(ests, str):
            # FIXME:
            if node1 == SOURCE_NODE:
                card1 = 1.0
            else:
                card1 = cards1[ests]

            if node2 == SOURCE_NODE:
                card2 = 1.0
            else:
                card2 = cards2[ests]
            card3 = cards3[ests]

        elif ests is None:
            # true costs
            card1 = cards1["actual"]
            card2 = cards2["actual"]
            card3 = cards3["actual"]
        else:
            assert isinstance(ests, dict)
            if node1 in ests:
                card1 = ests[node1]
                card2 = ests[node2]
                card3 = ests[node3]
            else:
                card1 = ests[" ".join(node1)]
                card2 = ests[" ".join(node2)]
                card3 = ests[" ".join(node3)]

        cost, edges_kind = get_costs(subset_graph, card1, card2, card3, node1,
                node2, cost_model)
        assert cost != 0.0
        subset_graph[edge[0]][edge[1]][cost_key] = cost
        subset_graph[edge[0]][edge[1]][cost_key + "scan_type"] = edges_kind

        total_cost += cost
    return total_cost

def get_costs(subset_graph, card1, card2, card3, node1, node2,
        cost_model):
    '''
    '''
    def update_edges_kind_with_seq(edges_kind, nilj_cost, cost2):
        if cost2 is not None and cost2 < nilj_cost:
            cost = cost2
            if len(node1) == 1:
                edges_kind["".join(node1)] = "Seq Scan"
            if len(node2) == 1:
                edges_kind["".join(node2)] = "Seq Scan"
        else:
            cost = nilj_cost
            if len(node1) == 1:
                edges_kind["".join(node1)] = "Index Scan"
                if len(node2) == 1:
                    edges_kind["".join(node2)] = "Seq Scan"
            elif len(node2) == 1:
                edges_kind["".join(node2)] = "Index Scan"
                if len(node1) == 1:
                    edges_kind["".join(node1)] = "Seq Scan"

    edges_kind = {}
    if cost_model == "C":
        # simple cost model for a left deep join
        if len(node1) == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif len(node2) == 1:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            assert False, "one of the nodes must have a single table"

        cost2 = card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost
        update_edges_kind_with_seq(edges_kind, nilj_cost, cost2)
    else:
        assert False, "cost model {} unknown".format(cost_model)

    return cost, edges_kind
