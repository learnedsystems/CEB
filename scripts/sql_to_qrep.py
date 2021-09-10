import os
import pdb
from networkx.readwrite import json_graph
from query_representation.utils import *
from query_representation.query import *

OUTPUT_DIR="./queries/joblight/all_joblight/"
INPUT_FN = "./queries/joblight.sql"
OUTPUT_FN_TMP = "{i}.sql"

make_dir(OUTPUT_DIR)

with open(INPUT_FN, "r") as f:
    data = f.read()

queries = data.split(";")
for i, sql in enumerate(queries):
    output_fn = OUTPUT_DIR + str(i+1) + ".pkl"
    if "SELECT" not in sql:
        continue

    qrep = parse_sql(sql, None, None, None, None, None,
            compute_ground_truth=False)

    qrep["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
    qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])

    save_qrep(output_fn, qrep)

