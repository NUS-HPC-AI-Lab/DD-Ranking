cd ..

wandb disabled
# wandb enabled
# wandb online

torchrun --nproc_per_node=1  classification/train.py \
    --model 'resnet18' \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_50ep'

torchrun --nproc_per_node=1  classification/train.py \
    --model 'resnet18' \
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
    --output-dir 'save/rn18_100ep'

torchrun --nproc_per_node=1  classification/train.py \
    --model 'resnet18' \
    --batch-size 256 \
    --epochs 200 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_200ep'