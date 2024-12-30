cd ..

# log file: log/sre2l_tiny_4k_ipc50_rn18.log
python classification/train_kd.py \
    --model 'resnet18' \
    --teacher-model 'resnet18' \
    --teacher-path '/path/to/resnet18_E50/checkpoint.pth' \
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
    --syn-data-path '/path/to/sre2l_tiny_rn18_4k_ipc100' \
    -T 20 \
    --image-per-class 50 \
    --output-dir 'save_kd/T18S18_T20_[4K].ipc_50'


# log file: log/sre2l_tiny_4k_ipc100_rn18.log
python classification/train_kd.py \
    --model 'resnet18' \
    --teacher-model 'resnet18' \
    --teacher-path '/path/to/resnet18_E50/checkpoint.pth' \
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
    --syn-data-path '/path/to/sre2l_tiny_rn18_4k_ipc100' \
    -T 20 \
    --image-per-class 100 \
    --output-dir 'save_kd/T18S18_T20_[4K].ipc_100'