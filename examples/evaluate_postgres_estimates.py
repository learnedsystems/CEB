import sys
sys.path.append(".")
from query_representation.query import *
from losses.losses import *
import glob
import argparse
import random

def main():
    qreps = []
    preds = []

    fns = list(glob.glob(args.query_dir + "/*"))

    all_qfns = []
    for qi,qdir in enumerate(fns):
        template_name = os.path.basename(qdir)
        if args.query_templates != "all":
            query_templates = args.query_templates.split(",")
            if template_name not in query_templates:
                print("skipping template ", template_name)
                continue

        # let's first select all the qfns we are going to load
        qfns = list(glob.glob(qdir+"/*.pkl"))
        all_qfns += qfns

    # some templates take significantly longer to compute postgres plan costs
    # --- hence, by shuffling the list, things run faster.
    random.shuffle(all_qfns)
    for qfn in all_qfns:
        qrep = load_qrep(qfn)
        qreps.append(qrep)
        ests = get_postgres_cardinalities(qrep)
        preds.append(ests)

    print("going to call ppc for {} queries".format(len(qreps)))
    losses = compute_postgres_plan_cost(qreps, preds, port=args.port, num_processes=4,
            result_dir = args.result_dir)
    print(losses)

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", type=str, required=False,
            default="./queries/imdb/")
    parser.add_argument("--port", type=int, required=False,
            default=5432)
    parser.add_argument("--result_dir", type=str, required=False,
            default=None)
    parser.add_argument("--query_templates", type=str, required=False,
            default="all")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_flags()
    main()
