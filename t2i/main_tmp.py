import json
import os
import datetime
from tqdm import trange, tqdm

import PIL
import matplotlib.pyplot as plt

import torch
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T

# for dataloader
from transformers import BertTokenizer, BertModel
from transformers import CLIPTokenizer, CLIPTextModel


from models_tmp import Text2ImageModel 

class ImageTextDataset(Dataset):
    def __init__(self, root_dir="data/MSCOCO", split="train", image_size=256, t2i=True, model_name=None):
        self.split = split
        self.image_dir = f'data/MSCOCO/{split}2014'
        self.t2i = t2i

        ann = json.load(open(os.path.join(root_dir, "annotations", f'captions_{split}2014.json')))
        self.images = {item["id"]: item["file_name"] for item in ann["images"]} if t2i else [item["file_name"] for item in ann["images"]]
        self.texts = ann["annotations"]

        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.RandomResizedCrop(image_size, scale=(0.75, 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])
        if model_name == 'bert':
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif model_name == 'clip':
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        if self.t2i:
            return len(self.texts)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.t2i: # stage 2
            text = self.texts[idx]["caption"]
            path = os.path.join(self.image_dir, self.images[self.texts[idx]["image_id"]])
            image = self.image_transform(PIL.Image.open(path))
            text = get_embedding(text, self.model, self.tokenizer) # str -> tensor
        
        else: # stage1
            text = torch.empty(1, dtype=torch.int8) # no use
            path = os.path.join(self.image_dir, self.images[idx])
            image = self.image_transform(PIL.Image.open(path))
            
        return text, image


def main():
    # setup
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    experiment_name = 'overfit4'
    t2i = bool(1) # stage1: False, stage2: True
    mode = 't2i' if t2i else 'i2i'
    LM_name = 'bert' # ['bert': 768, 'clip': 512]
    text_embdim = 768

    # vae
    vae_name = 'vqgan_vae' # ['dvae', 'vqgan_vae']
    if vae_name == 'dvae':
        vae = OpenAIDiscreteVAE().to(device)
    elif vae_name == 'vqgan_vae':
        vae = VQGanVAE().to(device)

    # attention types
    attn_group = ['axial_row', 'axial_row', 'axial_row', 'axial_col'] # row:col = 3:1
    attn_types = attn_group * 3 + ['conv_like']

    # main model
    if t2i: # stage 2
        model = Text2ImageModel(dim=512, depth=13, vae=vae, ff_dropout=.0, t2i=t2i, text_embdim=text_embdim, attn_types=attn_types).to(device) 
        model.load_state_dict(torch.load(f'saved/checkpoints/{vae_name}_{experiment_name}_i2i.pt')['model'], strict=False)
        model.projection_layer = torch.nn.Linear(text_embdim, model.dim, device=device)
    else: # stage 1
        model = Text2ImageModel(dim=512, depth=13, vae=vae, ff_dropout=.0, t2i=t2i, attn_types=attn_types).to(device) 


    # data
    train_dataset, val_dataset = ImageTextDataset(split="train", t2i=t2i, model_name=LM_name), ImageTextDataset(split="val", t2i=t2i,  model_name=LM_name)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4) 
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # optimizer
    init_lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=0.)
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.001, total_iters=warmup_epochs)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.7, verbose=True, threshold=.5, patience=10)

    # training
    best_train_loss, best_val_loss = torch.inf, torch.inf
    for epoch in range(1, 101):
        
        print(f'epoch {epoch} @ {datetime.datetime.now()}')
        current_scheduler = warmup_scheduler if epoch <= 1+warmup_epochs else plateau_scheduler
        if epoch <= 1+warmup_epochs:
            print('lr =', current_scheduler.get_last_lr()[0]) # print warmup lr
        train_epoch_loss, val_epoch_loss = 0, 0

        model.train()
        for batch in (train_dataloader): 
            optimizer.zero_grad()
            texts, images = batch[0].to(device), batch[1].to(device)
            loss = model(text=texts, image=images, return_loss=True)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.detach().item()

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                texts, images = batch[0].to(device), batch[1].to(device)
                loss = model(text=texts, image=images, return_loss=True)
                
                val_epoch_loss += loss.detach().item()

            texts, images = map(lambda x: x.to(device), next(iter(val_dataloader)))
            if epoch == 1: # only save original images once
                save_original_image(images)
                pass

            if is_best_model(val_epoch_loss, best_val_loss): # save the model and images with best val loss
                save_model(model, optimizer, val_epoch_loss, current_scheduler, epoch, vae_name, mode, experiment_name)
                best_val_loss = val_epoch_loss
                if epoch >= 1+warmup_epochs: # save time
                    save_generated_image(model, vae_name, epoch, texts, mode, experiment_name)
                    pass
        
        current_scheduler.step(None if epoch <= 1+warmup_epochs else val_epoch_loss) # pass in val_epoch_loss to plateau optimizer

        print(f'training loss: {train_epoch_loss/len(train_dataloader)}')
        print(f'val loss: {val_epoch_loss/len(val_dataloader)}')
        print('-------------------------------------')

		
# helpers

@torch.no_grad()
def get_embedding(text: str, model, tokenizer):
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    outputs = model(**inputs)
    pooled_output = outputs.pooler_output  # pooled (EOS token) states
    return pooled_output # (1, dim=768/512 for bert/clip)

def is_best_model(loss, best_loss):
    return loss < best_loss

def save_original_image(images):
    ori_img = torchvision.utils.make_grid(images, nrow=8, normalize=True, range=(images.min(), images.max()))
    ori_img_pth = os.path.join('saved', f'original.png')
    plt.imsave(ori_img_pth, ori_img.permute(1,2,0).cpu().numpy())
    print(f'Original images are saved to {ori_img_pth}.')


def save_generated_image(model, vae_name, epoch, texts, mode, experiment_name):
    generated_images = model.generate_images(text=texts, use_cache=True)
    rec_img = torchvision.utils.make_grid(generated_images, \
        nrow=4, normalize=True, range=(generated_images.min(), generated_images.max()))
    if not os.path.isdir(f'saved/{experiment_name}'):
        os.mkdir(f'saved/{experiment_name}')
    
    rec_img_pth = os.path.join('saved', experiment_name, f'generated_{vae_name}_{experiment_name}_{mode}_E{epoch}.png')
    plt.imsave(rec_img_pth, rec_img.permute(1,2,0).cpu().numpy()) # c h w -> h w c
    print(f'Generated {vae_name} images saved at epoch {epoch} in {rec_img_pth}.')


def save_model(model, optimizer, loss, scheduler, epoch, vae_name, mode, experiment_name):
    d = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss, 'scheduler': scheduler.state_dict()}
    model_pth = f'saved/checkpoints/{vae_name}_{experiment_name}_{mode}.pt'
    torch.save(d, model_pth)
    print(f'{vae_name} model saved at epoch {epoch} in {model_pth}.')


if __name__ == '__main__':
    main()
