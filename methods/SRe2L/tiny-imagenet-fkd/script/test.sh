cd ..

wandb disabled

for ckpt in ./save/rn18_50ep/checkpoint.pth \
            ./save/rn18_100ep/checkpoint.pth \
            ./save/rn18_200ep/checkpoint.pth; do

    echo $ckpt
    torchrun --nproc_per_node=1 classification/train.py \
        --model 'resnet18' \
        --batch-size 256 \
        --test-only \
        --resume $ckpt

done

for ckpt in ./save/rn50_50ep/checkpoint.pth \
            ./save/rn50_100ep/checkpoint.pth \
            ./save/rn50_200ep/checkpoint.pth; do

    echo $ckpt
    torchrun --nproc_per_node=1 classification/train.py \
        --model 'resnet50' \
        --batch-size 256 \
        --test-only \
        --resume $ckpt

done