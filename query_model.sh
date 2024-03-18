MODEL='google/t5-xl-lm-adapt'

python query_model.py \
    --model ${MODEL} \
    --match_type 'exact' \
    --verbose 1