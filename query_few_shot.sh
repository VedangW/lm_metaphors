MODEL='google/t5-xl-lm-adapt'

python few_shot.py \
    --model ${MODEL} \
    --device 0 \
    --verbose 1