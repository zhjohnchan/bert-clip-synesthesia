import json

import evaluate
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, CLIPTextModelWithProjection


@torch.no_grad()
def main(model_name, batch_size=64):
    # 1. Preprocess the data
    cxc_path = "data/cxc/sts_val.csv"
    caption_path = "data/cxc/captions_val2014.json"
    cxc_ann = pd.read_csv(cxc_path)
    caption_ann = json.load(open(caption_path))
    sid2caption = {}
    sid2iid = {}
    for sample in caption_ann["annotations"]:
        sid2caption[sample["id"]] = sample["caption"]
        sid2iid[sample["id"]] = sample["image_id"]
    caption_1 = []
    caption_2 = []
    sts_l_score = []
    sts_v_score = []
    for row_id, row in cxc_ann.iterrows():
        caption_1.append(sid2caption[int(row.caption1.split(":")[-1])])
        caption_2.append(sid2caption[int(row.caption2.split(":")[-1])])
        sts_l_score.append(row.agg_score)
        sts_v_score.append(1 if sid2iid[int(row.caption1.split(":")[-1])] == sid2iid[int(row.caption2.split(":")[-1])]
                           else 0)
    ann = pd.DataFrame({"caption_1": caption_1, "caption_2": caption_2,
                        "sts_l_score": sts_l_score, "sts_v_score": sts_v_score})

    # 2. Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "clip" in model_name:
        model = CLIPTextModelWithProjection.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    model.to('cuda')
    model.eval()
    similarities = []

    # 3. Compute the similarities
    for i in tqdm(range(len(ann) // batch_size + 1)):
        caption_1 = ann.caption_1.iloc[i * batch_size:(i + 1) * batch_size].tolist()
        caption_2 = ann.caption_2.iloc[i * batch_size:(i + 1) * batch_size].tolist()

        inputs_1 = tokenizer(caption_1, padding=True, truncation=True, return_tensors="pt").to('cuda')
        inputs_2 = tokenizer(caption_2, padding=True, truncation=True, return_tensors="pt").to('cuda')

        outputs_1 = model(**inputs_1)
        outputs_2 = model(**inputs_2)

        if "clip" in model_name:
            similarities.extend(F.cosine_similarity(outputs_1[0], outputs_2[0], dim=-1))
        else:
            similarities.extend(F.cosine_similarity(outputs_1[1], outputs_2[1], dim=-1))

    # 4. Compute the metrics
    results = {}
    pearsonr_metric = evaluate.load("pearsonr")
    spearmanr_metric = evaluate.load("spearmanr")
    for gt_name, gt_score in zip(["sts-l", "sts-v"], [sts_l_score, sts_v_score]):
        pearsonr = pearsonr_metric.compute(predictions=similarities, references=gt_score)
        spearmanr = spearmanr_metric.compute(references=similarities, predictions=gt_score)
        results[f'pearsonr_{model_name}_{gt_name}'] = pearsonr
        results[f'spearmanr_{model_name}_{gt_name}'] = spearmanr

    return results


if __name__ == '__main__':
    model_names = [
        "bert-base-cased", "bert-large-cased",
        "roberta-base", "roberta-large",
        "princeton-nlp/unsup-simcse-bert-base-uncased", "princeton-nlp/unsup-simcse-bert-large-uncased",
        "princeton-nlp/unsup-simcse-roberta-base", "princeton-nlp/unsup-simcse-roberta-large",
        "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14-336"
    ]
    rsts = []
    for name in model_names:
        rst = main(name)
        rsts.append(rst)
        print(rst)
