base_dir="/output/FMAE_det"
save_dir="/model/zhujy/IMDL/FMAE_det"
mkdir -p ${base_dir}
mkdir -p ${save_dir}

# CUDA_VISIBLE_DEVICES=0 \
# torchrun  \
#     --standalone    \
#     --nnodes=1     \
    
python \
./test.py \
    --model FMAE \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_json "./test_datasets_det.json" \
    --checkpoint_path ${save_dir} \
    --test_batch_size 1 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --alpha 0.15 \
    --beta 0.35 \
    --gamma 0.5 \
    --if_detect