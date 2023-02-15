CUDA_VISIBLE_DEVICES=2 python ../src/clip_fine_tune.py \
                --dataset 'FashionIQ' \
                --num-epochs 100 \
                --clip-model-name RN50 \
                --encoder both \
                --learning-rate 2e-6 \
                --batch-size 128 --transform targetpad --target-ratio 1.25 \
                --save-training --save-best --validation-frequency 1