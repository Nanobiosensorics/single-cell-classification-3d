#!/usr/bin/bash

SRC_PATH=$1
DATA_TYPE=$2
KFOLD=$3

declare -a arr=("fibronectin" "noncoated")
declare -a times=(30 60 90 120)
declare -a cell_types=(hela mdamb231 mcf7 lclc hepg2 h838)

for i in "${arr[@]}"
do
    python src/create_dataset.py -p $SRC_PATH/$i/3d-coating-evaluation/ -d $DATA_TYPE -t ${times[@]} -c ${cell_types[@]} -k $KFOLD
done