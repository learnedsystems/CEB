import pickle
import pdb
from collections import defaultdict
import numpy as np

STATS_CARD_FN = "/flash1/pari/End-to-End-CardEst-Benchmark/workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"
QREP_DIR_FMT = "./queries/stats2/all_stats2/{i}.pkl"

# CARDS_FMT = "mscn_stats_final_results/MSCN780559495-simplestats-ours/queries/stats2-preds/{i}.pkl"
# CARDS_FMT = "stats_results/MSCN3922184360/queries/stats2-preds/{i}.pkl"
# CARDS_FMT = "stats_results/MSCN780559495/queries/stats2-preds/{i}.pkl"
#CARDS_FMT = "./mscn_stats_final_results/MSCN406889911-simplestats-default/queries/stats2-preds/{i}.pkl"
# CARDS_FMT ="mscn_stats_final_results/MSCN1032811392/queries/stats2-preds/{i}.pkl"
# CARDS_FMT ="mscn_stats_final_results/MSCN1737532302/queries/stats2-preds/{i}.pkl"
CARDS_FMT ="stats_results/MSCN876113495/queries/stats2-preds/{i}.pkl"


with open(STATS_CARD_FN, "r") as f:
    data = f.readlines()

query_our_cards = defaultdict(list)
query_cards = defaultdict(list)

our_ests = []
true_ests = []

for li, line in enumerate(data):
    splits = line.split("||")
    query = splits[0]
    assplits = query.split("as")

    aliases = []
    for ass in assplits:
        ass = ass.lower()
        if "from" in ass:
            continue

        if "where" in ass.lower():
            ass = ass[0:ass.find("where")]
        else:
            ass = ass[0:ass.find(",")]

        ass = ass.replace(" ", "")
        aliases.append(ass)

    qnum = int(splits[1])
    truecard = splits[2]

    qrepfn = QREP_DIR_FMT.format(i = li+1)
    cardfn = CARDS_FMT.format(i=li+1)

    with open(cardfn, "rb") as f:
        cards = pickle.load(f)
    aliases.sort()
    qkey = tuple(aliases)
    assert qkey in cards

    our_ests.append(cards[qkey])
    true_ests.append(int(truecard))

    ## for later debugging
    query_cards[li].append([aliases, truecard, query])
    if li not in query_our_cards:
        query_our_cards[li].append(cards)

our_ests = np.array(our_ests)
true_ests = np.array(true_ests)
qerrs = np.maximum(our_ests/true_ests, true_ests/our_ests)
print("QErr mean: ", np.mean(qerrs), "; QErr median: ", np.median(qerrs),
        "; QErr: 90p: ", np.percentile(qerrs,90),  "; QErr 99p: ", np.percentile(qerrs, 99))

out_fn = CARDS_FMT.format(i = "all")
out_fn = out_fn.replace(".pkl", ".txt")

print(out_fn)
np.savetxt(out_fn, our_ests, delimiter='\n', fmt='%1.4f')

total = 0
ourtotal = 0

for qnum in query_cards:
    truedata = query_cards[qnum]
    ourcards = query_our_cards[qnum][0]

    total += len(truedata)
    ourtotal += len(ourcards)

    # trueqs = [t[2] for t in truedata]

print("Total: ", total, "; Our total: ", ourtotal)
