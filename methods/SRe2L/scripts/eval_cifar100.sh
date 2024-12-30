GPU=$0

CUDA_VISIBLE_DEVICES=${GPU} python ./small_dataset/squeeze_cifar100.py \
    --epochs 200 \
    --output-dir ./save/cifar100/resnet18_E200 \

CUDA_VISIBLE_DEVICES=${GPU} python ./small_dataset/recover_cifar.py \
    --arch-name "resnet18" \
    --arch-path 'save/cifar100/resnet18_E200/ckpt.pth' \
    --exp-name "cifar100_rn18_1K_mobile.lr0.25.bn0.01" \
    --batch-size 100 \
    --lr 0.25 \
    --iteration 1000 \
    --r-bn 0.01 \
    --num_classes 100 \
    --store-best-images \
    --ipc-start 0 \
    --ipc-end 50 \

CUDA_VISIBLE_DEVICES=${GPU} python ./small_dataset/relabel_cifar100.py \
    --epochs 400 \
    --output-dir ./save_post_cifar100/ipc50 \
    --syn-data-path syn_data/cifar100_rn18_1K_mobile.lr0.25.bn0.01/50 \
    --teacher-path save/cifar100/resnet18_E200/ckpt.pth \
    --ipc 50 \
    --batch-size 128