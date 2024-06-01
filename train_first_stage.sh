DATA_PATH="./data/kitti"
exp=$1
model_name=$2
GPU_NUM=$3
BS=$4
PY_ARGS=${@:5}

EXP_DIR=./log/$exp
LOG_DIR=$EXP_DIR/$model_name
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

# for single gpu
# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --dataset kitti \
#     --data_path $DATA_PATH \
#     --log_dir $EXP_DIR  \
#     --model_name $model_name \
#     --split eigen_zhou \
#     --height 192 \
#     --width 640 \
#     --png \
#     --batch_size $BS \
#     --num_workers 12 \
#     --learning_rate 5e-5 \
#     --num_layers 18 \

# for multi gpus
# first stage training 
export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.launch --nproc_per_node $GPU_NUM --master_port 2556 -m train \
    --dataset kitti \
    --data_path $DATA_PATH \
    --log_dir $EXP_DIR  \
    --model_name $model_name \
    --split eigen_zhou \
    --height 192 \
    --width 640 \
    --png \
    --batch_size $BS \
    --num_workers 12 \
    --learning_rate 5e-5 \
    --num_layers 18 \
    --num_epochs 20 \
    --ddp \
    --aug_fp \
    $PY_ARGS | tee -a $EXP_DIR/$model_name/log_train.txt    

