TASK="$1"
TRAIN_SET="$2"
GPU="$3"

python script/experiment/train.py \
--task ${TASK} \
-d ${GPU} \
--train_set ${TRAIN_SET} \
--test_sets ${TRAIN_SET} \
--epochs_per_val 10