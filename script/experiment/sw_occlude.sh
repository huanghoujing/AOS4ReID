TRAIN_SET="$1"
GPU="$2"

python script/experiment/sw_occlude.py \
-d ${GPU} \
--train_set ${TRAIN_SET}