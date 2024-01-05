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


# second stage training
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node $GPU_NUM --master_port 2549 -m train_ts \
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
    --aug_fp \
    --ddp \
    --pseudo_weight 1 \
    --stable_thre 0.04 \
    --depth_thre 0.04 \
    --ema_weight 0.85 \
    --load_weights_folder ./log/train/first_stage_model/models/weights_19 \
    --models_to_load encoder depth pose_encoder pose \
    $PY_ARGS | tee -a $EXP_DIR/$model_name/log_train.txt   