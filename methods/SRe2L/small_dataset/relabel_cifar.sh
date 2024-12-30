CUDA_VISIBLE_DEVICES=6 \
python relabel_cifar.py \
    --epochs 400 \
    --output-dir ./save_post_cifar10/ipc50 \
    --syn-data-path syn_data/cifar10_rn18_1K_mobile.lr0.25.bn0.01/50 \
    --teacher-path save/cifar10/resnet18_E200/ckpt.pth \
    --ipc 50 --batch-size 128
