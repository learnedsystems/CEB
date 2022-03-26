ALG=mscn
NORMFL=1
#LR=0.0001
HLS=128
#QDIR=queries/imdb
REPS=(1 1 1)
SEED=0

ONEHOTD=0
TF=1
JF=onehot-stats
PF=onehot-stats
FLOWF=1
EVALE=200
EPOCHS=10

for r in "${!REPS[@]}";
do
    bash run_all_diff_flow2.sh $TF $JF $PF $NORMFL $HLS $FLOWF $ONEHOTD $EVALE $EPOCHS
done
