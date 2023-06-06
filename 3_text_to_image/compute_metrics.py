import os
import sys

import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def clip_score(paths, texts, batch_size=128):
    assert len(paths) == len(texts)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    model.to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    scores = []
    for i in tqdm(range(len(paths) // batch_size)):
        images = [Image.open(path) for path in paths[i * batch_size:(i + 1) * batch_size]]

        inputs = processor(text=texts[i * batch_size:(i + 1) * batch_size], images=images,
                           return_tensors="pt", padding=True).to("cuda")
        outputs = model(**inputs)
        scores.extend(torch.diag(outputs.logits_per_image).cpu().tolist())
    return sum(scores) / len(scores)


@torch.no_grad()
def main(record_path, batch_size=128):
    _ = torch.manual_seed(123)

    real_paths, fake_paths, texts = [], [], []
    fin = open(record_path)
    for line_idx, line in enumerate(fin):
        real_paths.append(line.split("\t")[0])
        fake_paths.append(line.split("\t")[1])
        texts.append(line.split("\t")[2])

    real_images = torch.cat([ToTensor()(Resize((256, 256))(Image.open(path))).unsqueeze(0).to(torch.uint8)
                             for path in real_paths], dim=0)
    fake_images = torch.cat([ToTensor()(Image.open(path)).unsqueeze(0).to(torch.uint8)
                             for path in fake_paths], dim=0)

    metric_clip_score = clip_score(fake_paths, texts, batch_size)
    print(f'metric_clip_score: {metric_clip_score}')
    metric_clip_score_real = clip_score(real_paths, texts, batch_size)
    print(f'metric_clip_score_real: {metric_clip_score_real}')

    inception = InceptionScore()
    for i in range(len(real_images) // batch_size):
        inception.update(fake_images[i * batch_size:(i + 1) * batch_size])
    metric_is = inception.compute()
    print(f'metric_is: {metric_is}')

    fid = FrechetInceptionDistance()
    for i in range(len(real_images) // batch_size):
        fid.update(real_images[i * batch_size:(i + 1) * batch_size], real=True)
        fid.update(fake_images[i * batch_size:(i + 1) * batch_size], real=False)
    metric_fid = fid.compute()
    print(f'metric_fid: {metric_fid}')

    return {"is": metric_is,
            "fid": metric_fid,
            "clip_score": metric_clip_score,
            "clip_score_real": metric_clip_score_real}


if __name__ == '__main__':
    # record_path = "generated_images/outputs_bert-base-uncased_last.ckpt/records.txt"
    metrics = main(sys.argv[1])
    print(metrics)
    fout = open(os.path.join(os.path.dirname(sys.argv[1]), "metrics.txt"), "wt")
    fout.write(str(metrics))
