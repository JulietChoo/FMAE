base_dir="/model/zhujy/IMDL/FMAE_loc"
mkdir -p ${base_dir}

python \
./test.py \
    --model FMAE \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_json "./test_datasets_loc.json" \
    --checkpoint_path ${base_dir} \
    --test_batch_size 1 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --alpha 0.3 \
    --beta 0.7 \
    --gamma 0