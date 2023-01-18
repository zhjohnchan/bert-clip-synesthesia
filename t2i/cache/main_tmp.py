import json
import os
import datetime
from typing import Iterable
from tqdm import trange, tqdm

import PIL
import matplotlib.pyplot as plt

import torch
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T


from models_tmp import Text2ImageModel # dbg

class ImageTextDataset(Dataset):
    def __init__(self, root_dir="data/MSCOCO", split="train", image_size=256, t2i=True):
        self.split = split
        self.image_dir = f'data/MSCOCO/{split}2014'
        self.t2i = t2i

        ann = json.load(open(os.path.join(root_dir, "annotations", f'captions_{split}2014.json')))
        self.images = {item["id"]: item["file_name"] for item in ann["images"]} if t2i else [item["file_name"] for item in ann["images"]] # dbg
        self.texts = ann["annotations"] # dbg

        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.RandomResizedCrop(image_size, scale=(0.75, 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])
        

    def __len__(self):
        if self.t2i:
            return len(self.texts)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.t2i:
            text = self.texts[idx]["caption"]
            path = os.path.join(self.image_dir, self.images[self.texts[idx]["image_id"]])
            image = self.image_transform(PIL.Image.open(path))
            
        
        else: # train transformer only
            text = torch.empty(1, dtype=torch.int8) # no use
            path = os.path.join(self.image_dir, self.images[idx])
            image = self.image_transform(PIL.Image.open(path))
            
        return text, image


def main():
    # setup
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    i2i_experiment_name = 'overfit3'
    t2i_experiment_name = 'lr1e-2_E20_clip' 
    epoch = 20 # loading i2i models 
    bs = 128
    mode = 't2i' 
    LM_name = 'clip' # ['bert': 768, 'clip': 512] 
    text_embdim = 512 if LM_name == 'clip' else 768
    rotary_emb = True # if True, use rotary embedding, else use Axial Embedding

    t2i = mode=='t2i' # stage1: False, stage2: True

    # vae
    vae_name = 'vqgan_vae' # ['dvae', 'vqgan_vae']
    if vae_name == 'dvae':
        vae = OpenAIDiscreteVAE().to(device)
    elif vae_name == 'vqgan_vae':
        vae = VQGanVAE().to(device)

    # attention types
    # attn_group = ['axial_row', 'axial_col', 'axial_row', 'axial_row'] # row:col = 3:1
    # attn_types = attn_group * 3 + ['conv_like'] # combination of sparse attentions
    attn_types = None # full attention

    # main model
    if t2i: # stage 2, load a model from stage 1 and add projection layer
        model = Text2ImageModel(
			dim=512, 
			depth=13, 
			vae=vae, 
			ff_dropout=.0, 
			t2i=t2i, 
			text_embdim=text_embdim, 
			attn_types=attn_types, 
			rotary_emb=rotary_emb, 
			LM_name=LM_name).to(device) 
        model_pth = f'saved/{i2i_experiment_name}/{vae_name}_{i2i_experiment_name}_i2i_E{epoch}.pt'
        print(f'loading from {model_pth}')
        model.load_state_dict(torch.load(model_pth)['model'], strict=False)
        model.projection_layer = torch.nn.Sequential( # add projection layer
            torch.nn.Linear(text_embdim, model.dim, device=device),
            torch.nn.Dropout(0.3)
        )
    else: # stage 1
        model = Text2ImageModel(dim=512, depth=13, vae=vae, ff_dropout=.0, t2i=t2i, attn_types=attn_types, rotary_emb=rotary_emb).to(device) # dbg


    # data
    train_dataset, val_dataset = ImageTextDataset(split="train", t2i=t2i), ImageTextDataset(split="val", t2i=t2i)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=5) 
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=5)

    # optimizer
    init_lr = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=4.5e-2) 
    warmup_epochs = 4
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.001, total_iters=warmup_epochs)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.7, verbose=True, threshold=.5, patience=5,
    ) 

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
            texts, images = batch[0], batch[1].to(device)
            loss = model(text=texts, image=images, return_loss=True)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.detach().item()

            

        model.eval()
        with torch.no_grad():
            for batch in (val_dataloader):
                texts, images = batch[0], batch[1].to(device)
                loss = model(text=texts, image=images, return_loss=True)
                
                val_epoch_loss += loss.detach().item()

            texts, images = next(iter(val_dataloader)) 
            texts, images = texts[:64], images[:64].to(device) # only generate 64 images
            if epoch == 1: # only save original images once
                save_original_image(images, vae_name, mode, i2i_experiment_name, t2i_experiment_name, LM_name)
                save_texts(texts, vae_name, mode, t2i_experiment_name, LM_name)
                pass

            if is_best_model(val_epoch_loss, best_val_loss): # save the model and images with best val loss
                save_model(model, optimizer, val_epoch_loss, current_scheduler, epoch, vae_name, mode, i2i_experiment_name, t2i_experiment_name, LM_name) 
                best_val_loss = val_epoch_loss
                if epoch >= 1+warmup_epochs: 
                    save_generated_image(model, vae_name, epoch, texts, mode, i2i_experiment_name, t2i_experiment_name, LM_name) # dbg
                    pass
        
        current_scheduler.step(None if epoch <= 1+warmup_epochs else val_epoch_loss/len(val_dataloader)) 

        print(f'training loss: {train_epoch_loss/len(train_dataloader)}')
        print(f'val loss: {val_epoch_loss/len(val_dataloader)}')
        print('-------------------------------------')


# helpers
def save_texts(texts: tuple, vae_name, mode, t2i_experiment_name, LM_name):
    if mode == 'i2i':
        return 
    pth = os.path.join(f'saved/{mode}/{t2i_experiment_name}', f'original_{vae_name}_{LM_name}_{t2i_experiment_name}_{mode}.txt')
    with open(pth, 'w') as f:
        for line in texts:
            f.write(line + '\n')
    print(f'Texts saved to {pth}')

def is_best_model(loss, best_loss):
    return loss < best_loss

def check_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def save_original_image(images, vae_name, mode, i2i_experiment_name, t2i_experiment_name, LM_name):
    ori_img = torchvision.utils.make_grid(images, nrow=8, normalize=True, range=(images.min(), images.max()))
    if mode == 'i2i':
        check_dir(f'saved/{i2i_experiment_name}')
        ori_img_pth = os.path.join('saved', i2i_experiment_name, f'original_{vae_name}_{i2i_experiment_name}_{mode}.png')
    else:
        check_dir(f'saved/{mode}/{t2i_experiment_name}')
        ori_img_pth = os.path.join(f'saved/{mode}/{t2i_experiment_name}', f'original_{vae_name}_{LM_name}_{t2i_experiment_name}_{mode}.png')
    plt.imsave(ori_img_pth, ori_img.permute(1,2,0).cpu().numpy())
    print(f'Original images are saved to {ori_img_pth}.')


def save_generated_image(model, vae_name, epoch, texts, mode, i2i_experiment_name, t2i_experiment_name, LM_name):
    generated_images = model.generate_images(text=texts, use_cache=True)
    rec_img = torchvision.utils.make_grid(generated_images, \
        nrow=8, normalize=True, range=(generated_images.min(), generated_images.max()))
    
    if mode == 'i2i':
        check_dir(f'saved/{i2i_experiment_name}')
        rec_img_pth = os.path.join('saved', i2i_experiment_name, f'generated_{vae_name}_{i2i_experiment_name}_{mode}_E{epoch}.png')
    else:
        check_dir(f'saved/{mode}/{t2i_experiment_name}')
        rec_img_pth = os.path.join(f'saved/{mode}/{t2i_experiment_name}', f'generated_{vae_name}_{LM_name}_{t2i_experiment_name}_{mode}_E{epoch}.png')
    plt.imsave(rec_img_pth, rec_img.permute(1,2,0).cpu().numpy()) # c h w -> h w c
    print(f'Generated {vae_name} images saved at epoch {epoch} in {rec_img_pth}.')


def save_model(model, optimizer, loss, scheduler, epoch, vae_name, mode, i2i_experiment_name, t2i_experiment_name, LM_name):
    if epoch % 2 != 0 or epoch < 10:
        return
    d = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss, 'scheduler': scheduler.state_dict()}
    if mode == 'i2i':
        check_dir(f'saved/{i2i_experiment_name}')
        model_pth = f'saved/{i2i_experiment_name}/{vae_name}_{i2i_experiment_name}_{mode}_E{epoch}.pt'
        torch.save(d, model_pth)
        print(f'{vae_name} model saved at epoch {epoch} in {model_pth}.')
    elif mode == 't2i':
        check_dir(f'saved/{mode}/{t2i_experiment_name}')
        model_pth = f'saved/{mode}/{t2i_experiment_name}/{vae_name}_{LM_name}_{t2i_experiment_name}_{mode}_E{epoch}.pt'
        torch.save(d, model_pth)
        print(f'{vae_name} model for {LM_name} saved at epoch {epoch} in {model_pth}.')



if __name__ == '__main__':
    main()
