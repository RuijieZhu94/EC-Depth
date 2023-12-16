export CUDA_VISIBLE_DEVICES=0
python evaluate_depth.py \
    --data_path ./data/kitti \
    --load_weights_folder ./weights_19 \
    --eval_mono \
    --height 192 \
    --width 640 \
    --png \
    --eval_split eigen \
    --batch_size 1 \






