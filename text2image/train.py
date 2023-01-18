import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from omegaconf import OmegaConf
from taming.data.base import ImagePaths
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from cond_transformer import Net2NetTransformer
from pytorch_lightning.loggers import CSVLogger


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image


class CelebAHQDataset(Dataset):
    def __init__(self, size, split="train"):
        super().__init__()
        root = "data/celebahq"
        split_path = "data/celebahqtrain.txt" if split == "train" else "data/celebahqvalidation.txt"
        with open(split_path, "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, f'{int(relpath[:-4])}.jpg') for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.texts = [t for relpath in relpaths
                      for t in open(os.path.join(f'{root}-caption', f'{int(relpath[:-4])}.txt')).read().splitlines()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        example = self.data[i // 10]
        example["text"] = self.texts[i]
        ex = example
        return ex


def load_model(text_model_name, learning_rate=0.0625):
    config_path = "2021-04-23T18-11-19_celebahq_transformer/configs/2021-04-23T18-11-19-project.yaml"
    ckpt_path = "2021-04-23T18-11-19_celebahq_transformer/checkpoints/last.ckpt"
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    config = OmegaConf.load(config_path)
    model = Net2NetTransformer(**config.model.params, learning_rate=learning_rate, text_model_name=text_model_name)
    missing, unexpected = model.load_state_dict(pl_sd["state_dict"], strict=False)
    return model


def main(text_model_name="bert-base-uncased"):
    # data
    train_dataset, val_dataset = CelebAHQDataset(256, split="train"), CelebAHQDataset(256, split="val")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    logger = CSVLogger("logs", name=f'{text_model_name}')

    # model
    model = load_model(text_model_name=text_model_name, learning_rate=1e-4)
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath=f'outputs/{text_model_name}/',
        filename='epoch{epoch:02d}-val_loss{val/loss:.2f}',
        auto_insert_metric_name=False,
        save_weights_only=True, save_last=True, save_top_k=1
    )

    # training
    trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16,
                         max_epochs=4, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main(text_model_name=sys.argv[1])
