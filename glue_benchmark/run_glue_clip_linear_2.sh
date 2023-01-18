tasks=(cola sst2 mrpc stsb qqp mnli qnli rte wnli)
epochs=(12 12 20 12 12 12 12 12 20)

for model in "openai/clip-vit-base-patch32" "openai/clip-vit-base-patch16" "openai/clip-vit-large-patch14" "openai/clip-vit-large-patch14-336"; do
  for i in "${!tasks[@]}"; do
    echo "${tasks[$i]}"
    python glue_benchmark/main.py \
    --model_name_or_path ${model} \
    --task_name "${tasks[$i]}" \
    --do_train \
    --do_eval \
    --linear_prob \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs "${epochs[$i]}" \
    --output_dir outputs_linear/${model}/"${tasks[$i]}"/ \
    --fp16
  done
done