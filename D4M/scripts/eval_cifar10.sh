GPU=$0

CUDA_VISIBLE_DEVICES=${GPU} python ./cifar_fkd/squeeze_cifar10.py \
    --epochs 200 \
    --output-dir ./save/cifar10/resnet18_E200 \


CUDA_VISIBLE_DEVICES=${GPU} python ./distillation/gen_prototype.py \
    --batch_size 10 \
    --data_dir ../../../datasets/CIFAR10 \
    --dataset cifar10 \
    --diffusion_checkpoints_path ../../../models/stable-diffusion-v1-5 \
    --ipc 50 \
    --km_expand 1 \
    --label_file_path ./distillation/label-prompt/CIFAR-10_labels.txt \
    --save_prototype_path ./prototypes \

CUDA_VISIBLE_DEVICES=${GPU} python ./distillation/gen_syn_image.py \
    --dataset cifar10 \
    --diffusion_checkpoints_path ../../../models/stable-diffusion-v1-5 \
    --guidance_scale 8 \
    --strength 0.7 \
    --ipc 50 \
    --km_expand 1 \
    --label_file_path ./distillation/label-prompt/CIFAR-10_labels.txt \
    --prototype_path ./prototypes/cifar10-ipc50-kmexpand1.json \
    --save_init_image_path ./distilled_data/ \

CUDA_VISIBLE_DEVICES=${GPU} python ./transform.py \
    --input_size 32 \
    --root_dir ./distilled_data/cifar10_ipc50_50_s0.7_g8.0_kmexpand1 \
    --save_dir ./resized_data/CIFAR10/IPC50 \

CUDA_VISIBLE_DEVICES=${GPU} python relabel_cifar.py \
    --epochs 400 \
    --output-dir ./save_post_cifar10/ipc50 \
    --syn-data-path ./resized_data/CIFAR10/IPC50 \
    --teacher-path ./save/cifar10/resnet18_E200/ckpt.pth \
    --ipc 50  \
    --batch-size 128 

