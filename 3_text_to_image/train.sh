export TOKENIZERS_PARALLELISM=true
mkdir -p logs/openai

# "bert-base-uncased", "bert-large-uncased",
# "roberta-base", "roberta-large",
# "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16",
# "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14-336"

model=bert-base-uncased
CUDA_VISIBLE_DEVICES=0 nohup python train.py ${model} > logs/${model}.log 2>&1 &

model=openai/clip-vit-base-patch32
CUDA_VISIBLE_DEVICES=1 nohup python train.py ${model} > logs/${model}.log 2>&1 &

model=roberta-base
CUDA_VISIBLE_DEVICES=2 nohup python train.py ${model} > logs/${model}.log 2>&1 &

model=openai/clip-vit-large-patch14
CUDA_VISIBLE_DEVICES=3 nohup python train.py ${model} > logs/${model}.log 2>&1 &

model=bert-large-uncased
CUDA_VISIBLE_DEVICES=0 nohup python train.py ${model} > logs/${model}.log 2>&1 &

model=openai/clip-vit-base-patch16
CUDA_VISIBLE_DEVICES=1 nohup python train.py ${model} > logs/${model}.log 2>&1 &

model=roberta-large
CUDA_VISIBLE_DEVICES=2 nohup python train.py ${model} > logs/${model}.log 2>&1 &

model=openai/clip-vit-large-patch14-336
CUDA_VISIBLE_DEVICES=3 nohup python train.py ${model} > logs/${model}.log 2>&1 &
