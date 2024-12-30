CUDA_VISIBLE_DEVICES=7 \
python recover_cifar.py \
--arch-name "resnet18" \
--arch-path 'save/cifar10/resnet18_E200/ckpt.pth' \
--exp-name "cifar10_rn18_1K_mobile.lr0.25.bn0.01" \
--batch-size 10 \
--lr 0.25 \
--iteration 1000 \
--r-bn 0.01 \
--store-best-images \
--ipc-start 0 --ipc-end 1

