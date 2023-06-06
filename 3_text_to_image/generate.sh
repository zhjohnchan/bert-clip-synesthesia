export TOKENIZERS_PARALLELISM=true

# "bert-base-uncased", "bert-large-uncased",
# "roberta-base", "roberta-large",
# "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16",
# "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14-336"

model=bert-base-uncased
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}
python compute_metrics.py generated_images/outputs_bert-base-uncased_last.ckpt/records.txt

model=openai/clip-vit-base-patch32
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}
python compute_metrics.py generated_images/outputs_openai_clip-vit-base-patch32_last.ckpt/records.txt

model=roberta-base
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}
python compute_metrics.py generated_images/outputs_roberta-base_last.ckpt/records.txt

model=openai/clip-vit-large-patch14
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}
python compute_metrics.py generated_images/outputs_openai_clip-vit-large-patch14_last.ckpt/records.txt

model=bert-large-uncased
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}
python compute_metrics.py generated_images/outputs_bert-large-uncased_last.ckpt/records.txt

model=openai/clip-vit-base-patch16
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}
python compute_metrics.py generated_images/outputs_openai_clip-vit-base-patch16_last.ckpt/records.txt

model=roberta-large
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}
python compute_metrics.py generated_images/outputs_roberta-large_last.ckpt/records.txt

model=openai/clip-vit-large-patch14-336
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}
python compute_metrics.py generated_images/outputs_openai_clip-vit-large-patch14-336_last.ckpt/records.txt

