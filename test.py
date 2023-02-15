import pickle

FN = "stats_results/MSCN4032526393-stats-simple-ours/queries/stats-preds/1.pkl"
with open(FN, "rb") as f:
    data = pickle.load(f)
print(data)
