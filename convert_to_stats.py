import pickle
import pdb
from collections import defaultdict

STATS_CARD_FN = "/flash1/pari/End-to-End-CardEst-Benchmark/workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"
QREP_DIR_FMT = "./queries/stats/all_stats/{i}.pkl"
# CARDS_FMT = "stats_results/MSCN4032526393-stats-simple-ours/queries/stats-preds/{i}.pkl"
CARDS_FMT = "mscn_stats_final_results/MSCN1782604579-simplestats-default/queries/stats2-preds/{i}.pkl"

with open(STATS_CARD_FN, "r") as f:
    data = f.readlines()

query_our_cards = defaultdict(list)
query_cards = defaultdict(list)

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

    qrepfn = QREP_DIR_FMT.format(i = qnum+1)
    cardfn = CARDS_FMT.format(i=qnum+1)

    with open(cardfn, "rb") as f:
        cards = pickle.load(f)

    # print(qnum+1)
    # print(aliases)
    # print(cards)
    # print(truecard)

    query_cards[qnum].append([aliases, truecard, query])

    if qnum not in query_our_cards:
        query_our_cards[qnum].append(cards)

total = 0
ourtotal = 0
for qnum in query_cards:
    truedata = query_cards[qnum]
    ourcards = query_our_cards[qnum][0]

    total += len(truedata)
    ourtotal += len(ourcards)

    trueqs = [t[2] for t in truedata]
    print(trueqs[-1])
    for trueq in trueqs:
        if len(trueq) > len(trueqs[-1]):
            print(qnum)
            print(trueqs[-1])
            print(trueq)
            pdb.set_trace()

    # pdb.set_trace()
    # if len(ourcards) < len(truedata):
        # print("we don't have enough cardinalities!")
        # print(len(ourcards), len(truedata))
        # print(ourcards)
        # print(truedata)
        # pdb.set_trace()

print("Total: ", total, "; Our total: ", ourtotal)
