#!/usr/bin/bash

start=`date +%s`

SRC_PATH=$1
RES_PATH=$2
TYPE=$3
KFOLD=$4

declare -a models=("cnn" "densenet" "resnet" "cnn-lstm")
declare -a times=(30 60 90)
declare -a cell_types=(hela mdamb231 mcf7 lclc hepg2 h838)

for model in "${models[@]}"
do
    for t in "${times[@]}"
    do
        python src/run.py -p $SRC_PATH -ct ${cell_types[@]} -tm $t -t $TYPE -r $RES_PATH -e 250 --normalize std -m $model -k $KFOLD
        python src/eval.py -p $SRC_PATH -ct ${cell_types[@]} -tm $t -t $TYPE -r $RES_PATH --normalize std -c ../../data/classification_datasets/ -m $model -k $KFOLD
    done
done

end=`date +%s`

echo $((end-start))