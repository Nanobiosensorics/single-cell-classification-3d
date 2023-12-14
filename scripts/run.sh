SRC_PATH=$1
KFOLD=$2

bash scripts/run_3d_class_preo.sh $SRC_PATH/fibronectin/3d-coating-evaluation/ ../../data/classification_datasets/total-results/fibronectin/3d-coating-evaluation/result/ im_cover 1;
bash scripts/run_3d_class_preo.sh $SRC_PATH/fibronectin/3d-coating-evaluation/ ../../data/classification_datasets/total-results/fibronectin/3d-coating-evaluation/result/ im_pred 1;
bash scripts/run_3d_class_preo.sh $SRC_PATH/fibronectin/3d-coating-evaluation/ ../../data/classification_datasets/total-results/fibronectin/3d-coating-evaluation/result/ im_watershed 1;

bash scripts/run_3d_class.sh $SRC_PATH/fibronectin/3d-coating-evaluation/ $SRC_PATH/fibronectin/3d-coating-evaluation/result/ im_cover $KFOLD;
bash scripts/run_3d_class.sh $SRC_PATH/fibronectin/3d-coating-evaluation/ $SRC_PATH/fibronectin/3d-coating-evaluation/result/ im_pred $KFOLD;
bash scripts/run_3d_class.sh $SRC_PATH/fibronectin/3d-coating-evaluation/ $SRC_PATH/fibronectin/3d-coating-evaluation/result/ im_watershed $KFOLD;

bash scripts/run_3d_class.sh $SRC_PATH/noncoated/3d-coating-evaluation/ $SRC_PATH/noncoated/3d-coating-evaluation/result/ im_cover $KFOLD;
bash scripts/run_3d_class.sh $SRC_PATH/noncoated/3d-coating-evaluation/ $SRC_PATH/noncoated/3d-coating-evaluation/result/ im_pred $KFOLD;
bash scripts/run_3d_class.sh $SRC_PATH/noncoated/3d-coating-evaluation/ $SRC_PATH/noncoated/3d-coating-evaluation/result/ im_watershed $KFOLD;