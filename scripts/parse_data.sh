#!/usr/bin/bash

python src/parse.py -p ../../data/classification_datasets/total-results/fibronectin/ -i ../../data/classification_datasets/fibronectin/ -d ../../data/classification_datasets/total-results/fibronectin/3d-coating-evaluation/ -c preo hela mdamb231 mcf7 lclc hepg2 h838
python src/parse.py -p ../../data/classification_datasets/total-results/noncoated/ -i ../../data/classification_datasets/noncoated/ -d ../../data/classification_datasets/total-results/noncoated/3d-coating-evaluation/ -c preo hela mdamb231 mcf7 lclc hepg2 h838

# python src/parse.py -p ../../data/classification_datasets/non-adjacent-results/fibronectin/ -i ../../data/classification_datasets/fibronectin/ -d ../../data/classification_datasets/non-adjacent-results/fibronectin/3d-coating-evaluation/ -c preo hela mdamb231 mcf7 lclc hepg2 h838
# python src/parse.py -p ../../data/classification_datasets/non-adjacent-results/noncoated/ -i ../../data/classification_datasets/noncoated/ -d ../../data/classification_datasets/non-adjacent-results/noncoated/3d-coating-evaluation/ -c preo hela mdamb231 mcf7 lclc hepg2 h838