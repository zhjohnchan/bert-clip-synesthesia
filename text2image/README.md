# Text-to-Image Generation

## Step 1: Install taming-transformers

```angular2html
cd taming-transformers
pip install -e .
```

# Step 2: Training

```angular2html
model=bert-base-uncased
python train.py ${model}
```

# Step 3: Generation

```angular2html
python generate.py
```
