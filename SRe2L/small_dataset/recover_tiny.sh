CUDA_VISIBLE_DEVICES=6 \
python recover_tiny.py \
--arch-name "resnet18" \
--arch-path './save/TinyImageNet/checkpoint.pth' \
--exp-name "sre2l_tiny_rn18_4k" \
--syn-data-path './syn_data' \
--batch-size 200 \
--lr 0.1 \
--r-bn 1 \
--iteration 4000 \
--store-last-images \
--ipc-start 0 --ipc-end 50

