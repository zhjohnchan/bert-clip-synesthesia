import json
import os
import types
import datetime
from typing import Iterable
from tqdm import trange, tqdm

import PIL
import numpy as np
import matplotlib.pyplot as plt

import torch
from taming.models.cond_transformer import Net2NetTransformer
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T

from transformers import BertTokenizer, BertModel
from transformers import CLIPTokenizer, CLIPTextModel

from omegaconf import OmegaConf
from einops import rearrange



class ImageTextDataset(Dataset):
    def __init__(self, root_dir="data/MSCOCO", split="train", image_size=256,):
        self.split = split
        self.image_dir = f'data/MSCOCO/{split}2014'

        ann = json.load(open(os.path.join(root_dir, "annotations", f'captions_{split}2014.json')))
        self.images = {item["id"]: item["file_name"] for item in ann["images"]} 
        self.texts = ann["annotations"] # dbg

        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.RandomResizedCrop(image_size, scale=(0.75, 1.), ratio=(1., 1.)),
            T.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]["caption"]
        path = os.path.join(self.image_dir, self.images[self.texts[idx]["image_id"]])
        image = self.image_transform(PIL.Image.open(path))
        
        return text, image


def main():
    # setup
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    experiment_name = 'pilot0'

    bs = 20 # dbg
    LM_name = 'bert'

    if LM_name == 'bert':
        text_embdim = 768
        LM = BertModel.from_pretrained("bert-base-uncased").to('cpu')
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif LM_name == 'clip':
        text_embdim = 512
        LM = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to('cpu')
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


    # model
    config, model = laod_config_and_model(device)
    dim = config.model.params.transformer_config.params.n_embd
    model.projection_layer = torch.nn.Sequential( # add projection layer
        torch.nn.Linear(text_embdim, dim),
        # torch.nn.LayerNorm(dim),
        # torch.nn.Tanh(),
        # torch.nn.Linear(dim, dim),
    ).to(device)

    # override forward method of transformer. The new `forward` is defined below.
    model.transformer.forward = types.MethodType(forward, model.transformer) 

    # freeze everything except the linear layer
    for name, p in model.named_parameters():
        if 'projection_layer' in name:
            p = p.requires_grad_(True)
        else:
            p = p.requires_grad_(False)

    def init_weights(m: torch.nn.Module):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    model.projection_layer.apply(init_weights)


    # data
    train_dataset, val_dataset = ImageTextDataset(split="train"), ImageTextDataset(split="val")
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=5) 
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=5)

    # optimizer
    init_lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=4.5e-2) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, threshold=0.5)
    # print(sum(p.numel() for p in model.parameters())); exit()

    # training
    best_train_loss, best_val_loss = torch.inf, torch.inf
    for epoch in range(1, 101):
        
        print(f'epoch {epoch} @ {datetime.datetime.now()}')
        
        train_epoch_loss, val_epoch_loss = 0, 0

        model.train()
        for batch in (train_dataloader): 
            optimizer.zero_grad()
            texts, images = batch[0], batch[1].to(device)
            texts = get_embedding(texts, LM, tokenizer, device) 
            texts = model.projection_layer(texts) # (bs, seqlen, dim)
            img_emb, images = model.encode_to_z(images) # (bs, seqlen=256, 16, 16), (bs, seqlen=256)
            logits, loss = model.transformer(idx=images[:, :-1], embeddings=texts, targets=images)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.detach().item()

            
        model.eval()
        with torch.no_grad():
            for batch in (val_dataloader):
                texts, images = batch[0], batch[1].to(device)
                texts = get_embedding(texts, LM, tokenizer, device)
                texts = model.projection_layer(texts)
                img_emb, images = model.encode_to_z(images)
                logits, loss = model.transformer(idx=images[:, :-1], embeddings=texts, targets=images)

                val_epoch_loss += loss.detach().item()

            texts, images = next(iter(val_dataloader)) 
            texts, images = texts[:64], images[:64].to(device) # only generate 64 images
            save_texts(texts, LM_name, experiment_name, epoch)
            
            texts = get_embedding(texts, LM, tokenizer, device)
            texts = model.projection_layer(texts)
            # generate images with texts prompts
            img_emb, _ = model.encode_to_z(images)
            z_shape = img_emb.shape
            generated_images = generate_images(model, texts, z_shape) 

            save_original_image(images, LM_name, experiment_name, epoch)
            save_generated_image(generated_images, LM_name, experiment_name, epoch)
        
        scheduler.step(val_epoch_loss/len(val_dataloader)) 
        print(f'training loss: {train_epoch_loss/len(train_dataloader)}')
        print(f'val loss: {val_epoch_loss/len(val_dataloader)}')
        print('-------------------------------------')



def forward(self, idx=None, embeddings=None, targets=None):
    # forward the GPT model
    image_seqlen = idx.shape[1] if idx is not None else 0
    text_seqlen = embeddings.shape[1] if embeddings is not None else 0
    if idx is not None: # at training, or from the second step on at inference.
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        if embeddings is not None: # prepend word embeddings
            assert text_seqlen <= 256, 'too many text tokens'
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1) 
    else: # at the first step of inference, there is no idx (image tokens) provided.
        assert embeddings is not None
        token_embeddings = embeddings

    t = token_embeddings.shape[1]
    assert t <= self.block_size, "Cannot forward, model block size is exhausted."
    token_embeddings += self.pos_emb[:, :t, :] # add position embedding
    
    x = self.drop(token_embeddings)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.head(x) # (bs, seqlen, vocab_size)

    # if we are given some desired targets, calculate the loss
    loss = None
    if targets is not None:
        # at training, image_seqlen == 255
        loss = torch.nn.functional.cross_entropy(
            rearrange(logits[:, -(image_seqlen+1):, :], 'b s v -> b v s'), 
            targets
        )

    return logits, loss

@torch.no_grad()
def generate_images(model: Net2NetTransformer, texts: torch.FloatTensor, z_shape, temperature=1.0, top_k=100):
    # texts: projected embedding, (bs, seqlen, dim)
    print('generating images')
    image = None
    for cur_len in trange(256):
        logits, _ = model.transformer(embeddings=texts, idx=image)
        logits = logits[:, -1]/temperature # select the last one

        if top_k is not None:
            logits = model.top_k_logits(logits, top_k)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1) # sample from probs
        if image is None:
            image = idx
        else:
            image = torch.cat([image, idx], dim=1)
    generated_images = model.decode_to_img(rearrange(image, 'b (h w) -> b h w', h=16), z_shape)

    return generated_images

# helpers
def laod_config_and_model(device):
    config_path = "sflckr/config.yaml"
    config = OmegaConf.load(config_path)

    model = Net2NetTransformer(**config.model.params)

    ckpt_path = "sflckr/last.ckpt"
    sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print('missing keys:', missing)
    if unexpected:
        print('unexpected keys:', unexpected)

    return config, model.to(device)

@torch.no_grad()
def get_embedding(text, model, tokenizer, device):
    if not isinstance(text, Iterable):
        text = [text]
    # inputs = tokenizer(text, padding=True, return_tensors="pt") # pad to same length within batch
    inputs = tokenizer(text, max_length=256, padding='max_length', return_tensors="pt") # pad to max_length
    
    outputs = model(**inputs)
    # pooled_output = outputs.pooler_output # (bs, dim), pooled (EOS token) states
    outputs = outputs.last_hidden_state # (bs, seqlen, dim)
    return outputs.to(device) #  dim=768 for bert, 512 for clip

def check_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def save_texts(texts: tuple, LM_name, experiment_name, epoch):
    if epoch != 1:
        return
    check_dir(f'saved/{experiment_name}')
    pth = os.path.join(f'saved/{experiment_name}', f'original_{LM_name}_{experiment_name}.txt')
    with open(pth, 'w') as f:
        for line in texts:
            f.write(line + '\n')
    print(f'Texts saved to {pth}')


def save_original_image(images, LM_name, experiment_name, epoch):
    if epoch !=1:
        return
    ori_img = torchvision.utils.make_grid(images, nrow=8, normalize=True, range=(images.min(), images.max()))
    
    check_dir(f'saved/{experiment_name}')
    ori_img_pth = os.path.join(f'saved/{experiment_name}', f'original_{LM_name}_{experiment_name}.png')
    plt.imsave(ori_img_pth, ori_img.permute(1,2,0).cpu().numpy())
    print(f'Original images are saved to {ori_img_pth}.')


def save_generated_image(generated_images, LM_name, experiment_name, epoch):
    
    rec_img = torchvision.utils.make_grid(generated_images, \
        nrow=8, normalize=True, range=(generated_images.min(), generated_images.max()))
    
    check_dir(f'saved/{experiment_name}')
    rec_img_pth = os.path.join(f'saved/{experiment_name}', f'generated_{LM_name}_{experiment_name}_E{epoch}.png')
    plt.imsave(rec_img_pth, rec_img.permute(1,2,0).cpu().numpy()) # c h w -> h w c
    print(f'Images prompted by {LM_name} at epoch {epoch} saved  in {rec_img_pth}.')


if __name__ == '__main__':
    main()
