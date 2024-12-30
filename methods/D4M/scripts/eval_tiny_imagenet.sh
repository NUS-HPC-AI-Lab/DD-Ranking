GPU=$0

CUDA_VISIBLE_DEVICES=${GPU} python ./distillation/gen_prototype.py \
    --batch_size 10 \
    --data_dir ../../../datasets/tiny-imagenet-200 \
    --dataset tiny_imagenet \
    --diffusion_checkpoints_path ../../../models/stable-diffusion-v1-5 \
    --ipc 50 \
    --km_expand 1 \
    --label_file_path ./distillation/label-prompt/tinyimgnt-label.txt \
    --save_prototype_path ./prototypes \

CUDA_VISIBLE_DEVICES=${GPU} python ./distillation/gen_syn_image.py \
    --dataset tiny_imagenet \
    --diffusion_checkpoints_path ../../../models/stable-diffusion-v1-5 \
    --guidance_scale 8 \
    --strength 0.7 \
    --ipc 50 \
    --km_expand 1 \
    --label_file_path ./distillation/label-prompt/tinyimgnt-label.txt \
    --prototype_path ./prototypes/tiny_imagenet-ipc50-kmexpand1.json \
    --save_init_image_path ./distilled_data/ \

CUDA_VISIBLE_DEVICES=${GPU} python ./transform.py \
    --input_size 64 \
    --root_dir ./distilled_data/tiny_imagenet_ipc50_50_s0.7_g8.0_kmexpand1 \
    --save_dir ./resized_data/TinyImageNet/IPC50 \
    
CUDA_VISIBLE_DEVICES=${GPU} python ./tiny_imagenet_fkd/classification/train_kd.py \
    --model 'resnet18' \
    --teacher-model 'resnet18' \
    --teacher-path './path/to/resnet18_E50/checkpoint.pth' \
    --batch-size 256 \
    --epochs 100 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --syn-data-path './resized_data/TinyImageNet/IPC50' \
    -T 20 \
    --image-per-class 50 \
    --output-dir './save_kd/T18_S18_T20_tiny.ipc_50'

