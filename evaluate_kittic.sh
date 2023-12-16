export CUDA_VISIBLE_DEVICES=0
python evaluate_depth_kittic.py \
    --data_path ./data \
    --load_weights_folder ./final_weight \
    --eval_mono \
    --height 192 \
    --width 640 \
    --png \
    --eval_split eigen \
    --batch_size 1 \


