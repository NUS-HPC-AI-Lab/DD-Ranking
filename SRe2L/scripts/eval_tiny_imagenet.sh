GPU=$0

CUDA_VISIBLE_DEVICES=${GPU} python ./small_dataset/recover_tiny.py \
    --arch-name "resnet18" \
    --arch-path './save/TinyImageNet/checkpoint.pth' \
    --exp-name "sre2l_tiny_rn18_4k" \
    --syn-data-path './syn_data' \
    --batch-size 200 \
    --lr 0.1 \
    --r-bn 1 \
    --iteration 4000 \
    --store-last-images \
    --ipc-start 0 \
    --ipc-end 50 \

CUDA_VISIBLE_DEVICES=${GPU} python ./tiny_imagenet_fkd/classification/train_kd.py \
    --model 'resnet18' \
    --teacher-model 'resnet18' \
    --teacher-path './save/TinyImageNet/checkpoint.pth' \
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
    --syn-data-path './syn_data/sre2l_tiny_rn18_4k/IPC50' \
    -T 20 \
    --image-per-class 50 \
    --output-dir './save_post_tiny/ipc50'