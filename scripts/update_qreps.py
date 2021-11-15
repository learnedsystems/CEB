import sys
import os
import pdb
sys.path.append(".")
import pickle
# from db_utils.utils import *
# from db_utils.query_storage import *
from query_representation.utils import *
from query_representation.query import *
import time

inp_dir = sys.argv[1]

# TODO: movie_companies and others
FUNCTIONAL_DEPS = {}
FUNCTIONAL_DEPS["k.keyword"] = "k.id"
FUNCTIONAL_DEPS["kt.kind"] = "kt.id"
FUNCTIONAL_DEPS["rt.role"] = "rt.id"

PKtoFK = {}
PKtoFK["k.id"] = "mk.keyword_id"
PKtoFK["kt.id"] = "t.kind_id"
PKtoFK["rt.id"] = "ci.role_id"

for tdir in os.listdir(inp_dir):
    if "1" not in tdir:
        continue
    start = time.time()
    fns = os.listdir(inp_dir + "/" + tdir)

    for fn in fns:
        qfn = inp_dir + "/" + tdir + "/" + fn
        qrep = load_qrep(qfn)
        print(qrep["join_graph"])
        pdb.set_trace()

        # save_qrep(qfn, qrep)

    print("saved all the queries with updated totals for ", tdir)
    print("took: ", time.time() - start)
