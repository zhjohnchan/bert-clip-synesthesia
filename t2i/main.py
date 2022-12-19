import json
import os

import PIL
import torch
from dalle_pytorch import OpenAIDiscreteVAE
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from models import Text2ImageModel


class ImageTextDataset(Dataset):
    def __init__(self, root_dir="data/MSCOCO", split="train", image_size=256):
        self.split = split
        self.image_dir = f'data/MSCOCO/{split}2014'

        ann = json.load(open(os.path.join(root_dir, "annotations", f'captions_{split}2014.json')))
        self.images = {item["id"]: item["file_name"] for item in ann["images"]}
        self.texts = ann["annotations"]

        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.RandomResizedCrop(image_size, scale=(0.75, 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]["caption"]
        path = os.path.join(self.image_dir, self.images[self.texts[idx]["image_id"]])
        image = self.image_transform(PIL.Image.open(path))

        return text, image


def main():
    # model
    vae = OpenAIDiscreteVAE()
    model = Text2ImageModel(dim=256, depth=6, vae=vae)

    # data
    train_dataset, val_dataset = ImageTextDataset(split="train"), ImageTextDataset(split="val")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

    # optimizer

    # training
    for epoch in range(30):
        model.train()
        # for batch in train_dataloader:
        #     texts, images = batch
        #     loss = model(text=texts, image=images, return_loss=True)

        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                texts, images = batch
                loss = model(text=texts, image=images, return_loss=True)
                generated_images = model.generate_images(text=texts)
                import pdb
                pdb.set_trace()


if __name__ == '__main__':
    main()
