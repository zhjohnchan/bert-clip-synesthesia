# On the Difference of BERT-style and CLIP-style Text Encoders
This is the official implementation of [On the Difference of BERT-style and CLIP-style Text Encoders](https://arxiv.org/abs/2306.03678) at Findings of ACL-2023.

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

## Experiment 1: GLUE Benchmarking

### 1.1: BERT-style Text Encoders

```angular2html
bash run_glue_bert.sh
```

### 1.2: CLIP-style Text Encoders

```angular2html
bash run_glue_clip.sh
```

## Experiment 2: Textual Similarity

### 2.1 Prepare the data

Download from [this link](https://github.com/google-research-datasets/Crisscrossed-Captions) and put them in
the `data/cxc` folder.

### 2.2 Run the textual similarity measure

```angular2html
python main.py
```

## Experiment 3: Text-to-Image Generation

### 3.1 Prepare the data

Download the dataset from [this link](https://drive.google.com/drive/folders/1eVrGKfkbw7bh9xPcX8HJa-qWQTD9aWvf) and put them in the `data/celebahq` and `data/celebahq-caption` folders.

### 3.2: Install taming-transformers

```angular2html
cd taming-transformers
pip install -e .
```

### 3.3: Training
```angular2html
bash train.sh
```

### 3.4: Generation and Compute the metrics

```angular2html
bash generate.sh
```

## Citations

If you use or extend our work, please cite our paper at Findings of ACL-2023.
```
@inproceedings{chen-acl-2023-synesthesia,
    title = "On the Difference of BERT-style and CLIP-style Text Encoders",
    author = "Chen, Zhihong and
    Chen, Guiming Hardy and
    Diao, Shizhe and
    Wan, Xiang and
    Wang, Benyou",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = july,
    year = "2023",
}
```
