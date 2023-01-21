# On the Difference of BERT-style and CLIP-style Text Encoders

## Requirements

* Python==3.8
* torch==1.12.1
* torchvision==0.13.1
* torchmetrics==0.10.0
* torch-fidelity==0.3.0
* pytorch-lightning==1.7.7
* transformers==4.26.0.dev0
* datasets==2.8.1.dev0
* evaluate==0.4.0

## Study 1: Glue Benchmarking

### 1.1: BERT-style Text Encoders

```angular2html
bash run_glue_bert.sh
```

### 1.2: CLIP-style Text Encoders

```angular2html
bash run_glue_clip.sh
```

## Study 2: Textual Similarity (STS-V and STS-L)

### 2.1 Prepare the data

Download from [this link](https://github.com/google-research-datasets/Crisscrossed-Captions) and put on
the `data/cxc` folder.

### 2.2 Run the textual similarity measure

```angular2html
python main.py
```

## Study 3: Text-to-Image Generation

### 3.1 Prepare the data

Download from [this link](https://drive.google.com/drive/folders/1eVrGKfkbw7bh9xPcX8HJa-qWQTD9aWvf) and put on
the `data/cxc` folder.

### 3.2: Install taming-transformers

```angular2html
cd taming-transformers
pip install -e .
```

### 3.3: Training

```angular2html
model=bert-base-uncased
python train.py ${model}
```

or

```angular2html
bash train.sh
```

### 3.4: Generation and Compute the metrics

```angular2html
bash generate.sh
```
