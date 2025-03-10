base_dir="/output/FMAE_det"
save_dir="/model/zhujy/IMDL/FMAE_det"
mkdir -p ${base_dir}
mkdir -p ${save_dir}

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
    --epochs 100 \
    --blr 1e-4 \
    --if_predict_label \
    --find_unused_parameters \
    --image_size 512 \
    --if_resizing \
    --min_lr 0 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --test_data_path /data/zhujy/IMDL/SampleDataset_DF_det_300 \
    --warmup_epochs 4 \
    --output_dir ${save_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 16 \
    --seed 42 \
    --test_period 1 \
    --layer_decay 0.75 \
    --test_batch_size 1 \
    --if_detect \
    --sample_number 12614 \
    --all_number 17660 \
    --alpha 0.15 \
    --beta 0.35 \
    --gamma 0.5 \
    --gradmask_path /model/zhujy/IMDL/gradmask.pth \
    --gradmask_ratio 0.3 ;
    
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
