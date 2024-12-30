GPU=$0

CUDA_VISIBLE_DEVICES=${GPU} python ./distillation/gen_prototype.py \
    --batch_size 10 \
    --data_dir ../../../datasets/CIFAR100 \
    --dataset cifar100 \
    --diffusion_checkpoints_path ../../../models/stable-diffusion-v1-5 \
    --ipc 50 \
    --km_expand 1 \
    --label_file_path ./distillation/label-prompt/new_order.txt \
    --save_prototype_path ./prototypes \

CUDA_VISIBLE_DEVICES=${GPU} python ./distillation/gen_syn_image.py \
    --dataset cifar100 \
    --diffusion_checkpoints_path ../../../models/stable-diffusion-v1-5 \
    --guidance_scale 8 \
    --strength 0.7 \
    --ipc 50 \
    --km_expand 1 \
    --label_file_path ./distillation/label-prompt/new_order.txt \
    --prototype_path ./prototypes/cifar100-ipc50-kmexpand1.json \
    --save_init_image_path ./distilled_data/ \

CUDA_VISIBLE_DEVICES=${GPU} python ./transform.py \
    --input_size 32 \
    --root_dir ./distilled_data/cifar100_ipc50_50_s0.7_g8.0_kmexpand1 \
    --save_dir ./resized_data/CIFAR100/IPC50 \

CUDA_VISIBLE_DEVICES=${GPU} python relabel_cifar.py \
    --epochs 400 \
    --output-dir ./save_post_cifar100/ipc50 \
    --syn-data-path ./resized_data/CIFAR100/IPC50 \
    --teacher-path ./save/cifar100/resnet18_E200/ckpt.pth \
    --ipc 50  \
    --batch-size 128 

