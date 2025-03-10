logger_dir='/output/FMAE_loc'
base_dir="/model/zhujy/IMDL/FMAE_loc"
mkdir -p ${base_dir}

# CUDA_VISIBLE_DEVICES=0 \
# torchrun  \
#     --standalone    \
#     --nnodes=1     \
#     --nproc_per_node=1 \

python \
./train.py \
    --model FMAE \
    --world_size 1 \
    --batch_size 8 \
    --data_path None \
    --epochs 200 \
    --blr 1e-3 \
    --if_predict_label \
    --find_unused_parameters \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --test_data_path /data/zhujy/IMDL/SampleDataset_DF_loc_100 \
    --warmup_epochs 4 \
    --output_dir ${base_dir}/ \
    --log_dir ${logger_dir}/ \
    --accum_iter 16 \
    --seed 42 \
    --test_period 1 \
    --layer_decay 0.75 \
    --test_batch_size 1 \
    --sample_number 5123 \
    --all_number 7172 \
    --alpha 0.3 \
    --beta 0.7 \
    --gamma 0 \
    --gradmask_path /model/zhujy/IMDL/gradmask.pth \
    --gradmask_ratio 0.3 ;
    
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
    --log_dir ${logger_dir}/ \
    --alpha 0.3 \
    --beta 0.7 \
    --gamma 0