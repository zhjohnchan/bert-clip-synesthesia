export TOKENIZERS_PARALLELISM=true
mkdir -p logs/openai

# "bert-base-uncased", "bert-large-uncased",
# "roberta-base", "roberta-large",
# "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16",
# "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14-336"

model=bert-base-uncased
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}

#model=openai/clip-vit-base-patch32
#ckpt=outputs/${model}/last.ckpt
#python generate.py ${model} ${ckpt}

model=roberta-base
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}

#model=openai/clip-vit-large-patch14
#ckpt=outputs/${model}/last.ckpt
#python generate.py ${model} ${ckpt}

model=bert-large-uncased
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}

model=openai/clip-vit-base-patch16
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}

model=roberta-large
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}

model=openai/clip-vit-large-patch14-336
ckpt=outputs/${model}/last.ckpt
python generate.py ${model} ${ckpt}

