ALG=mscn
#BUCKETS_LIKE=10
BUCKETS=(10)
DECAYS=(0.0)
#REPS=(1 1 1)
#REPS=(1 1 1)
REPS=(1 1 1)
SEED1=1
LOSS=mse
EVALE=200
DROP=0
TRUEP=0.8
HEURISTIC=1
DROP1=0.0
DROP2=0.0
DROP3=0.0
EPOCHS=10
QDIR=queries/imdb-unique-plans
BITMAP=0
LOADF=1
SEPL=0
TRUEB=0

for r in "${!REPS[@]}";
do
for i in "${!BUCKETS[@]}";
  do
    for j in "${!DECAYS[@]}";
    do
      bash run_all_diff.sh $ALG ${DECAYS[$j]} ${BUCKETS[$i]} $LOSS $SEED1 \
        $DROP $TRUEP $EVALE $EPOCHS $HEURISTIC $DROP1 $DROP2 $DROP3 $QDIR \
        $BITMAP $LOADF $SEPL $TRUEB
    done
  done
done
