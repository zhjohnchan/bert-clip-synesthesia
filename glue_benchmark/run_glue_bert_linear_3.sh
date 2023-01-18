tasks=(cola sst2 mrpc stsb qqp mnli qnli rte wnli)
epochs=(15 15 25 15 15 15 15 15 25)

for model in "bert-base-cased" "bert-large-cased" "roberta-base" "roberta-large" "princeton-nlp/unsup-simcse-bert-base-uncased" "princeton-nlp/unsup-simcse-bert-large-uncased" "princeton-nlp/unsup-simcse-roberta-base" "princeton-nlp/unsup-simcse-roberta-large"; do
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