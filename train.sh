CUDA_VISIBLE_DEVICES=0,1,2 python train.py --sample_size 90 \
            --batch_size 1 \
            --exp_name test \
            --train_dataset_path /path/to/train_dataset \
            --val_dataset_path /path/to/val_dataset \
            --devices 3 \
            --accumulate_grad_batches 4 \
            --every_n_train_steps 2000 \
            --val_check_interval 8000 \
            --warmup_steps 1000 \
            --max_steps 30000 \
            
            