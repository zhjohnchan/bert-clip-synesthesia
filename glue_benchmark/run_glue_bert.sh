tasks=(cola sst2 mrpc stsb qqp mnli qnli rte wnli)

for task in ${tasks[*]}; do
  echo ${task}
  python glue/main.py \
  --model_name_or_path bert-base-cased \
  --task_name "${task}" \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir output_bert_base_cased/"${task}"/
done
