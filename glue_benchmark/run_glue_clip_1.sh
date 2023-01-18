tasks=(cola sst2 mrpc stsb qqp mnli qnli rte wnli)
epochs=(3 3 5 3 3 3 3 3 5)

for model in "openai/clip-vit-base-patch32"; do
  for i in "${!tasks[@]}"; do
    echo "${tasks[$i]}"
    python glue_benchmark/main.py \
    --model_name_or_path ${model} \
    --task_name "${tasks[$i]}" \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs "${epochs[$i]}" \
    --output_dir outputs/${model}/"${tasks[$i]}"/ \
    --fp16
  done
done