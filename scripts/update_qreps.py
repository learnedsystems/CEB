import sys
import os
import pdb
sys.path.append(".")
import pickle
from query_representation.utils import *
from query_representation.query import *
import time
import psycopg2 as pg

inp_dir = sys.argv[1]

FDEP_TMP = """SELECT {COL1}, {COL2} FROM {TABLE}"""

# TODO: movie_companies and others
FUNCTIONAL_DEPS = {}
FUNCTIONAL_DEPS["k.keyword"] = "k.id"
FUNCTIONAL_DEPS["kt.kind"] = "kt.id"
FUNCTIONAL_DEPS["rt.role"] = "rt.id"
# FUNCTIONAL_DEPS["mi.info_type_id"] = "it.id"

PKtoFK = {}
PKtoFK["k.id"] = ("mk", "mk.keyword_id")
PKtoFK["kt.id"] = ("t", "t.kind_id")
PKtoFK["rt.id"] = ("ci", "ci.role_id")

# PKtoFK["it.id"] = ("mi", "mi.info_type_id")

USER="pari"
HOST="localhost"
PORT=5432
DBNAME="imdb"
PWD=""
con = pg.connect(user=USER, host=HOST, port=PORT,
        password=PWD, dbname=DBNAME)
cursor = con.cursor()

functional_mappings = {}

for tdir in os.listdir(inp_dir):
    if not os.path.isdir(inp_dir + "/" + tdir):
        continue
    # if "1a" not in tdir:
        # continue

    start = time.time()
    fns = os.listdir(inp_dir + "/" + tdir)

    for fn in fns:
        qfn = inp_dir + "/" + tdir + "/" + fn
        qrep = load_qrep(qfn)
        for node in qrep["join_graph"].nodes():
            nodedata = qrep["join_graph"].nodes()[node]
            for col in nodedata["pred_cols"]:
                col = ''.join([ck for ck in col if not ck.isdigit()])
                if col in functional_mappings:
                    continue
                if col in FUNCTIONAL_DEPS:
                    functional_mappings[col] = {}
                    table_name = nodedata["real_name"]
                    from_str = "{} as {}".format(table_name, node)
                    col2 = FUNCTIONAL_DEPS[col]
                    execcmd = FDEP_TMP.format(COL1=col,
                                              COL2 = col2,
                                              TABLE=from_str)
                    cursor.execute(execcmd)
                    exp_output = cursor.fetchall()
                    for (val1,val2) in exp_output:
                        functional_mappings[col][val1] = str(val2)

        for node in qrep["join_graph"].nodes():
            nodedata = qrep["join_graph"].nodes()[node]
            for ci, col in enumerate(nodedata["pred_cols"]):
                col = ''.join([ck for ck in col if not ck.isdigit()])
                if nodedata["pred_types"][ci] != "in":
                    continue

                if col in functional_mappings:
                    colid = FUNCTIONAL_DEPS[col]
                    colvals = nodedata["pred_vals"][ci]
                    cmap = functional_mappings[col]
                    (fknode, fkcol) = PKtoFK[colid]
                    node2data = qrep["join_graph"].nodes()[fknode]
                    newvals = []
                    for cv in colvals:
                        if cv in cmap:
                            newvals.append(cmap[cv])

                    node2data["implied_pred_vals"] = [newvals]
                    node2data["implied_pred_cols"] = [fkcol]
                    node2data["implied_pred_from"] = [col]
                    # pdb.set_trace()

        save_qrep(qfn, qrep)

    print("saved all the queries with updated qreps for ", tdir)
    print("took: ", time.time() - start)

con.close()
cursor.close()
