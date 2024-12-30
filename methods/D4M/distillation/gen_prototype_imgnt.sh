CUDA_VISIBLE_DEVICES=5 python gen_prototype.py \
    --batch_size 10 \
    --data_dir ../../datasets/tiny-imagenet-200 \
    --dataset tiny_imagenet \
    --diffusion_checkpoints_path ../../models/stable-diffusion-v1-5 \
    --ipc 50 \
    --km_expand 1 \
    --label_file_path ./label-prompt/tinyimgnt-label.txt \
    --save_prototype_path ./prototypes