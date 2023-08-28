from cardinality_estimation.fcnn import FCNN
from cardinality_estimation.mscn import MSCN, MSCN_JoinKeyCards
from cardinality_estimation.mstn import MSTN
from cardinality_estimation.algs import *

def get_alg(alg, cfg):
    if alg == "saved":
        return SavedPreds(model_dir=cfg["model_dir"])

    elif alg == "postgres":
        return Postgres()
    elif alg == "ms":
        return MSSQL(kind=alg)
    elif alg == "legms":
        return MSSQL(kind=alg)
    elif alg == "true":
        return TrueCardinalities()
    elif alg == "true_rank":
        return TrueRank()
    elif alg == "true_random":
        return TrueRandom()
    elif alg == "true_rank_tables":
        return TrueRankTables()
    elif alg == "joinkeys":
        return TrueJoinKeys()
    elif alg == "random":
        return Random()
    elif alg == "rf":
        return RandomForest(grid_search = False,
                n_estimators = 100,
                max_depth = 10,
                lr = 0.01)
    elif alg == "xgb":
        return XGBoost(grid_search=False, tree_method="hist",
                       subsample=1.0, n_estimators = 100,
                       max_depth=10, lr = 0.01)
    elif alg == "fcnn":
        return FCNN(
                cfg["model"],
                use_wandb = cfg["eval"]["use_wandb"],
                )
    elif alg == "mscn":
        return MSCN(
                cfg["model"],
                use_wandb = cfg["eval"]["use_wandb"],
                )

    elif alg == "mstn":
        return MSTN(
                cfg["model"],
                use_wandb = cfg["eval"]["use_wandb"])

    elif alg == "mscn_joinkey":
        return MSCN_JoinKeyCards(
                cfg["model"],
                use_wandb = cfg["eval"]["use_wandb"])

    else:
        assert False

