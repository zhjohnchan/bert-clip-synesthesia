import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal import CLIPScore
from torchvision.transforms import ToTensor, Resize


@torch.no_grad()
def main(record_path):
    _ = torch.manual_seed(123)

    real_paths, fake_paths, texts = [], [], []
    fin = open(record_path)
    for line in fin:
        real_paths.append(line.split("\t")[0])
        fake_paths.append(line.split("\t")[1])
        texts.append(line.split("\t")[2])

    real_images = torch.cat([ToTensor()(Resize((256, 256))(Image.open(path))).unsqueeze(0) for path in real_paths], dim=0)
    fake_images = torch.cat([ToTensor()(Image.open(path)).unsqueeze(0) for path in fake_paths], dim=0)

    inception = InceptionScore()
    inception.update(fake_images)
    metric_is = inception.compute()

    fid = FrechetInceptionDistance()
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    metric_fid = fid.compute()

    clip_score = CLIPScore()
    metric_clip_score = clip_score(fake_images, texts)

    return metric_is, metric_fid, metric_clip_score


if __name__ == '__main__':
    record_path = "generated_images/outputs_bert-base-uncased_last.ckpt/records.txt"
    metrics = main(record_path)
    print(metrics)
