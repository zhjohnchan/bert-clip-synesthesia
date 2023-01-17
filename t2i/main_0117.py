from argparse import ArgumentParser
import json
import logging
import os
from time import time
import types
import datetime
from typing import Iterable
from tqdm import trange, tqdm

import PIL
from numpy import mean, std
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal import CLIPScore

from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import BartTokenizer, BartModel
import clip
from BLIP.models.blip import blip_feature_extractor

from omegaconf import OmegaConf
from taming.models.cond_transformer import Net2NetTransformer
# pkgs: timm, fairscale, einops
# overwrite taming
torch.manual_seed(42)
def init_weights(m: torch.nn.Module):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
def get_args():
    parser = ArgumentParser()

    parser.add_argument("--lm", type=str, required=True) # 
    parser.add_argument('--mlp', action='store_true') # if true, use mlp

    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--tunelm', action='store_true')

    parser.add_argument('--generate', action='store_true') # load from a checkpoint and evaluate
    parser.add_argument('--use_cache', action='store_true')

    parser.add_argument("--local_rank", type=int, default=-1) # distributed learning
    parser.add_argument('--val', action='store_true')

    arg = parser.parse_args()
    
    # print(arg)
    return arg

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

        return text, image, torch.tensor(idx).long()

    def idx2str(self, idx: torch.LongTensor):
        text = tuple()
        idx = idx.detach().cpu().numpy().reshape(-1).tolist()
        for i in idx:
            text += tuple([self.texts[i]["caption"]])
        return text

def train(arg, model, train_dataloader, val_dataloader, train_sampler, \
    optimizer, scheduler, tokenizer, LM, LM_name, experiment_name, \
        device, n_samples, temperature, top_k, top_p, all_in_one, format):
    for epoch in range(1, 101):
        train_sampler.set_epoch(epoch) if arg.ddp else None # shuffle samples for different epochs
        logging.info(f'epoch {epoch} @ {datetime.datetime.now()}')
        
        train_epoch_loss, val_epoch_loss = 0, 0

        model.train() 
        for texts, images, _ in (train_dataloader): 
            optimizer.zero_grad() 
            texts, images = texts, images.to(device)
            texts = get_embedding(texts, LM, tokenizer, device)
            if arg.ddp:
                texts = model.module.projection_layer(texts)
                img_emb, images = model.module.encode_to_z(images) # (bs, seqlen=256, 16, 16), (bs, seqlen=256)
                logits, loss = model.module.transformer(idx=images[:, :-1], embeddings=texts, targets=images)
            else:
                texts = model.projection_layer(texts)
                img_emb, images = model.encode_to_z(images) # (bs, seqlen=256, 16, 16), (bs, seqlen=256)
                logits, loss = model.transformer(idx=images[:, :-1], embeddings=texts, targets=images)
            loss.backward()
            optimizer.step() 
            train_epoch_loss += loss.detach().item()
        scheduler.step(val_epoch_loss/len(val_dataloader))
            
        model.eval()
        with torch.no_grad():
            val_epoch_loss = save_everything(arg, val_dataloader, model, optimizer, scheduler, train_epoch_loss,\
                tokenizer, LM, LM_name, experiment_name, epoch, device, n_samples=n_samples, temperature=temperature, \
                    top_k=top_k, top_p=top_p, all_in_one=all_in_one, format=format) 
            
        
        logging.info(f'All items are saved.')
        torch.distributed.barrier() if torch.distributed.is_initialized() else None

        logging.info(f'training loss: {train_epoch_loss/len(train_dataloader)}')
        logging.info(f'val loss: {val_epoch_loss/len(val_dataloader)}') if arg.val else None
        logging.info('-'*50)

def main():
    arg = get_args()
    logging.basicConfig(level=logging.INFO if arg.local_rank in [-1, 0] else logging.WARN, format='%(message)s')
    if arg.ddp:
        torch.cuda.set_device(arg.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    # setup
    device = f"cuda:{max(arg.local_rank, 0)}" if torch.cuda.is_available() else 'cpu'
    LM_name = arg.lm # dbg!!!!,
    {
        'bert-base-uncased':            109482240, 
        'bert-large-uncased':           340. , 
        'roberta-base':                 124645632,
        'clip-vit-base-patch32 (clip)': 63165952, 
        'clip-vit-large-patch14':       123060480, 
        'clip-vit-large-patch14-336':   123060480,
        
        'clip-vit-base-patch16':        63165952, 
    }
    train_experiment_name = f'train_{LM_name}' # dbg, for train
    format = 'pdf'
    all_in_one = False
    bs = 24 # dbg, 20/24/62 for V/A40/A80, train_dataloader
    # for generate
    n_samples = 800
    ckpt_name = 'E{load_epoch}.ckpt'
    top_k = 100 # dbg, n_vocab = 16384
    temperature = 8.
    generate_experiment_name = f'generate_T{temperature}_k{top_k}_pN_{LM_name}' # dbg, for only generate
    compute_every = 10
    val_bs = 160 # dbg, 60 when --train on (V100), 56/160 --generate only on (V/A80)
    top_p = None

    middle_dir = f'train_{LM_name}'
    if arg.mlp:
        train_experiment_name += '_mlp'
        generate_experiment_name += '_mlp'
        middle_dir += '_mlp'
    if arg.tunelm:
        train_experiment_name += '_tunelm'
        generate_experiment_name += '_tunelm'
        middle_dir += '_tunelm'
    load_dir = os.path.join('saved', middle_dir, 'checkpoints')
    load_epochs = [2] # dbg

    if arg.train:
        experiment_name = train_experiment_name
    elif not arg.train and arg.generate:
        experiment_name = generate_experiment_name
    else:
        NotImplementedError

    if LM_name == 'bert-base-uncased':
        text_embdim = 768
        LM = BertModel.from_pretrained(LM_name).to(device)
        tokenizer = BertTokenizer.from_pretrained(LM_name)
        
    elif LM_name == 'bert-large-uncased':
        text_embdim = 1024
        LM = BertModel.from_pretrained(LM_name).to(device)
        tokenizer = BertTokenizer.from_pretrained(LM_name)
    
    elif LM_name == 'roberta-base':
        text_embdim = 768
        LM = RobertaModel.from_pretrained(LM_name).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(LM_name)

    elif LM_name in ['clip-vit-base-patch32','clip-vit-base-patch16']:
        text_embdim = 512
        LM = CLIPTextModel.from_pretrained(f"openai/{LM_name}").to(device)
        tokenizer = CLIPTokenizer.from_pretrained(f"openai/{LM_name}")

    elif LM_name in ['clip-vit-large-patch14', 'clip-vit-large-patch14-336']:
        text_embdim = 768
        LM = CLIPTextModel.from_pretrained(f"openai/{LM_name}").to(device)
        tokenizer = CLIPTokenizer.from_pretrained(f"openai/{LM_name}")

    elif LM_name == 'bart':
        text_embdim = 512 # ???
        LM = BartModel.from_pretrained('facebook/bart-base').to('cpu')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    elif LM_name == 'blip':
        text_embdim = 768
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
        LM = blip_feature_extractor(pretrained=model_url, image_size=224, vit='base').to('cpu')
        tokenizer = None
    # print(f'{LM_name}:', sum(p.numel() for p in LM.parameters())); exit()
    
    # data
    train_dataset, val_dataset = ImageTextDataset(split="train"), ImageTextDataset(split="val") # 414113, 202654
    train_sampler = None if not arg.ddp else torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = None if not arg.ddp else torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset, bs, shuffle=(train_sampler is None), num_workers=5, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=5, sampler=val_sampler)
    if arg.train:
        # main model
        
        model = laod_model(arg, text_embdim, None, LM, device) 
        
        # optimizer
        init_lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=4.5e-2) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, threshold=0.5) 
        # print(sum(p.numel() for p in model.parameters())); exit()
        # training
        best_train_loss, best_val_loss = torch.inf, torch.inf
        logging.info(f'n_train={len(train_dataset)}, n_val={len(val_dataset)}')
        logging.info(f'Training from scratch at {datetime.datetime.now()}...')
        if arg.ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # sync bn
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arg.local_rank]) # remember to pass device id!!!
    
        train(arg, model, train_dataloader, val_dataloader, train_sampler, \
            optimizer, scheduler, tokenizer, LM, LM_name, experiment_name, \
                device, n_samples, temperature, top_k, top_p, all_in_one, format)
    if not arg.train and arg.generate:
        for load_epoch in load_epochs:
            ckpt_path = os.path.join(load_dir, ckpt_name.format(load_epoch=load_epoch))
            model = laod_model(arg, text_embdim, ckpt_path, LM, device) 
            model.eval()
            if arg.ddp:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # sync bn
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arg.local_rank]) # remember to pass device id!!!
            save_everything(arg, val_dataloader, model, None, None, None,\
                    tokenizer, LM, LM_name, experiment_name, load_epoch, device, n_samples=n_samples, temperature=temperature, \
                        top_k=top_k, top_p=top_p, all_in_one=all_in_one, format=format, compute_every=compute_every) 
    
        
def laod_model(arg, text_embdim, ckpt_path=None, lm=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    config_path = "cin_transformer/configs/2021-04-03T19-39-50-project.yaml"
    config = OmegaConf.load(config_path)
    model = Net2NetTransformer(**config.model.params)

    if arg.train:
        ckpt_path = "cin_transformer/checkpoints/last.ckpt"
        sd = torch.load(ckpt_path, map_location=device)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        dim = config.model.params.transformer_config.params.n_embd
        if arg.mlp:
            model.projection_layer = torch.nn.Sequential( # add projection layer
                torch.nn.Linear(text_embdim, dim),
                torch.nn.GELU(),
                torch.nn.LayerNorm(dim),
                torch.nn.Linear(dim, dim),
            ).to(device)
        else:
            model.projection_layer = torch.nn.Sequential( # add projection layer
                torch.nn.Linear(text_embdim, dim),
            ).to(device)
        model.projection_layer.apply(init_weights)
        model.transformer.forward = types.MethodType(forward, model.transformer) # override forward method of transformer
        model.transformer.forward_with_past = types.MethodType(forward_with_past, model.transformer) # override forward method of transformer
        model.top_p_logits = types.MethodType(top_p_logits, model) # add model.top_p_logits()
        # freeze everything except the linear layer
        for name, p in model.named_parameters():
            if 'projection_layer' in name:
                p = p.requires_grad_(True)
            else:
                p = p.requires_grad_(False)

        # lm
        model.lm = lm
        if not arg.tunelm: # freezing lm 
            logging.info('Freezing language models...')
            for p in model.lm.parameters():
                p = p.requires_grad_(False)
        else: # tuning lm
            logging.info('Also tuning language models...')
        # print(next(model.lm.parameters()).requires_grad); exit()
        

    if not arg.train and arg.generate: # only generate
        assert ckpt_path is not None
        logging.info(f'loading from {ckpt_path}')
        dim = config.model.params.transformer_config.params.n_embd
        if arg.mlp:
            model.projection_layer = torch.nn.Sequential( # add projection layer
                torch.nn.Linear(text_embdim, dim),
                torch.nn.GELU(),
                torch.nn.LayerNorm(dim),
                torch.nn.Linear(dim, dim),
            ).to(device)
        else:
            model.projection_layer = torch.nn.Sequential( # add projection layer
                torch.nn.Linear(text_embdim, dim),
            ).to(device)
        model.lm = lm
        model.transformer.forward = types.MethodType(forward, model.transformer) # override forward method of transformer
        model.transformer.forward_with_past = types.MethodType(forward_with_past, model.transformer) # override forward_with_past method of transformer
        model.top_p_logits = types.MethodType(top_p_logits, model) # add model.top_p_logits()
        sd = torch.load(ckpt_path, map_location=device)["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        for name, p in model.named_parameters():
            p = p.requires_grad_(False)
        p = p.requires_grad_(True)
        # print(model.transformer.pos_emb.shape) # (1, 256, 1536)


    return model.to(device)


@torch.no_grad()
def get_embedding(text, model, tokenizer, device):
    if not isinstance(text, Iterable):
        text = [text]
    if tokenizer is None: # blip
        output = model(torch.empty(1), text, mode='text')[:, 0, :]
        return output.to(device) # dim=768

    inputs = tokenizer(text, padding=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    pooled_output = outputs.pooler_output  # pooled (EOS token) states
    return pooled_output.unsqueeze(1).to(device) # (bs, 1, dim), (768 for bert/blip; 512 for clip)


def forward(self, idx=None, embeddings=None, targets=None):
    # forward the GPT model
    if idx is not None:
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
    else:
        assert embeddings is not None
        token_embeddings = embeddings

    t = token_embeddings.shape[1] # (bs, seqlen, dim)
    assert t <= self.block_size, "Cannot forward, model block size is exhausted."
    position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
    x = self.drop(token_embeddings + position_embeddings)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.head(x)

    # if we are given some desired targets also calculate the loss
    loss = None
    if targets is not None:
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, loss


def forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None):
    # inference only
    assert not self.training
    if idx is not None:
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
    else:
        assert embeddings is not None
        token_embeddings = embeddings
    # token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector
    # if embeddings is not None:              # prepend explicit embeddings
    #     token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
    if past is not None:
        assert past_length is not None
        # past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
        past_shape = list(past.shape)
        # expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
        # assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
        # position_embeddings = self.pos_emb[:, :past_length, :]  # not `past_length`  each position maps to a (learnable) vector
    # else:
    position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :] 
    
    x = self.drop(token_embeddings + position_embeddings)
    presents = []  # accumulate over layers
    for i, block in enumerate(self.blocks):
        # modify L78 in /mntnfs/med_data5/guimingchen/anaconda3/envs/cgm/lib/python3.9/site-packages/taming/modules/transformer/mingpt.py
        x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
        presents.append(present)

    x = self.ln_f(x)
    logits = self.head(x)
    # if we are given some desired targets also calculate the loss
    loss = None
    if targets is not None:
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss, torch.stack(presents)  # _, _, (n_layer, 2, b, nh, 1, dim_head)

def top_p_logits(self, logits, p):
    prob = torch.nn.functional.softmax(logits, dim=-1)
    sorted_logits, sorted_indices = torch.sort(prob, descending=True)
    cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    prob[indices_to_remove] = float('-inf')
    return prob

@torch.no_grad()
def generate_images(arg, model: Net2NetTransformer, texts, z_shape, temperature=1.0, top_k=100, top_p=None):
    # texts: projected embedding, (bs, seqlen, dim)
    image = None
    cond_len = 1
    past = None
    model.module.transformer.training = False
    
    for cur_len in range(256):
        assert arg.ddp
        if arg.use_cache:
            logits, _, past = model.module.transformer.forward_with_past(idx=image, embeddings=texts, past=past, past_length=(cur_len+cond_len-1))
            # if past is None:
            #     past = [present]
            # else:
            #     past.append(present)
        else:
            logits, _ = model.module.transformer(embeddings=texts, idx=image)
        logits = logits[:, -1]/temperature # select the last one

        if top_k is not None:
            logits = model.module.top_k_logits(logits, top_k)
            probs = torch.nn.functional.softmax(logits, dim=-1)
        if top_p is not None:
            probs = model.module.top_p_logits(logits, top_p)
        
        idx = torch.multinomial(probs, num_samples=1) # sample from probs
        if image is None:
            image = idx
        else:
            image = torch.cat([image, idx], dim=1)
    generated_images = model.module.decode_to_img(image, z_shape) if arg.ddp else model.decode_to_img(image, z_shape) 

    return generated_images

  
  
# -----------helpers-----------
def check_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

@torch.no_grad()
def save_everything(arg, val_dataloader, model, optimizer, scheduler, train_epoch_loss, tokenizer, LM, LM_name, experiment_name, epoch, device, n_samples, temperature=1.0, top_k=100, top_p=None, all_in_one=False, format='pdf', compute_every = 10):
    val_epoch_loss = 0 if arg.val else None
    if arg.local_rank in [0, -1]:
        save_model(arg, model, optimizer, scheduler, train_epoch_loss, val_epoch_loss, LM_name, experiment_name, epoch)
    
    
    n_computed = 0
    fid = FrechetInceptionDistance(feature=64).to(device)
    inception = InceptionScore().to(device)
    clip_scores = []
    if arg.val:
        logging.info(f'val starts @ {datetime.datetime.now()}')
    if arg.generate:
        logging.info(f'generating images starts @ {datetime.datetime.now()}')
        logging.info('using cache') if arg.use_cache else None
    start_time = time()
    
    for batch_idx, (texts, images, idx) in (enumerate(val_dataloader, start=1)):
        if arg.generate or arg.val:
            texts, images = texts, images.to(device)
            texts = get_embedding(texts, LM, tokenizer, device)
            img_emb, encoded_images = model.module.encode_to_z(images)
            texts = model.module.projection_layer(texts)
            if arg.val:
                # img_emb, encoded_images = model.module.encode_to_z(images)
                # generate images with texts prompts
                logits, loss = model.module.transformer(idx=encoded_images[:, :-1], embeddings=texts, targets=encoded_images)
                val_epoch_loss += loss.detach().item() # update epoch loss

            if arg.generate:
                z_shape = img_emb.shape
                cur_batch_generated_images = generate_images(arg, model, texts, z_shape, temperature=temperature, top_k=top_k, top_p=top_p)
                cur_batch_indices, cur_batch_original_images = idx.to(device), images
        else:
            return None
        # if i == 0: # store all samples on current gpu
        #     indices = cur_batch_indices.detach().cpu()
        #     original_images = cur_batch_original_images.detach().cpu()
        #     generated_images = cur_batch_generated_images.detach().cpu() if not arg.train_only else None
            
        # else:
        #     indices = torch.cat([indices, cur_batch_indices.detach().cpu()])
        #     original_images = torch.cat([original_images, cur_batch_original_images.detach().cpu()])
        #     generated_images = torch.cat([generated_images, cur_batch_generated_images.detach().cpu()]) if not arg.train_only else None
            

        assert arg.ddp
        if arg.generate:
            # gather all samples and update metrics
            world_size = torch.distributed.get_world_size()

            # gather samples from all gpus
            cur_gathered_indices = [torch.zeros_like(cur_batch_indices).to(cur_batch_indices) for _ in range(world_size)]
            torch.distributed.all_gather(cur_gathered_indices, cur_batch_indices)
            cur_gathered_indices = torch.cat(cur_gathered_indices)
            cur_gathered_texts = val_dataloader.dataset.idx2str(cur_gathered_indices)
            
            cur_gathered_original_images = [torch.zeros_like(cur_batch_original_images).to(cur_batch_original_images) for _ in range(world_size)]
            torch.distributed.all_gather(cur_gathered_original_images, cur_batch_original_images)
            cur_gathered_original_images = torch.cat(cur_gathered_original_images)
            
            cur_gathered_generated_images = [torch.zeros_like(cur_batch_generated_images).to(cur_batch_generated_images) for _ in range(world_size)]
            torch.distributed.all_gather(cur_gathered_generated_images, cur_batch_generated_images)
            cur_gathered_generated_images = torch.cat(cur_gathered_generated_images)
        
            # update
            save = n_computed < n_samples
            n_computed += len(cur_gathered_generated_images)
            fid_pipeline(fid, cur_gathered_original_images, cur_gathered_generated_images, batch_idx, compute_every, n_computed) 
            inception_pipeline(inception, cur_gathered_generated_images, batch_idx, compute_every, n_computed)
            clip_pipeline(clip_scores, cur_gathered_generated_images, cur_gathered_texts, batch_idx, compute_every, n_computed, device=device)
            if arg.local_rank in [-1, 0] and save:
                save_original_image(cur_gathered_original_images, cur_gathered_indices.detach_().to('cpu').numpy().tolist(), LM_name, experiment_name, epoch, all_in_one, format)
                save_generated_image(cur_gathered_generated_images, cur_gathered_indices.detach_().to('cpu').numpy().tolist(), LM_name, experiment_name, epoch, all_in_one, format)
            # store samples from all gpus
            # if batch_idx == 1:
                # all_gathered_indices = cur_gathered_indices.detach_().to('cpu')
                # all_gathered_original_images = cur_gathered_original_images.detach_().to('cpu')
                # all_gathered_generated_images = None if not arg.generate else cur_gathered_generated_images.detach_().to('cpu') 
            # else:
                # all_gathered_indices = torch.cat([all_gathered_indices, cur_gathered_indices.detach_().to('cpu')])
                # all_gathered_original_images = torch.cat([all_gathered_original_images, cur_gathered_original_images.detach_().to('cpu')])
                # all_gathered_generated_images = None if not arg.generate else torch.cat([all_gathered_generated_images, cur_gathered_generated_images.detach_().to('cpu')])


    if arg.val:
        logging.info(f'val ends @ {datetime.datetime.now()}')
    if arg.generate:
        logging.info(f'generating images ends @ {datetime.datetime.now()}. {time() - start_time}s/{n_computed}={(time()-start_time)/n_computed}s per sample..')
        fid_score = fid.compute().item()
        logging.info('*'*20)
        logging.info(f'Metrics for all samples:')
        logging.info(f'FID: {fid_score}')
        fid.cpu()
        del fid

        inception_score_mean, inception_score_std = inception.compute()
        logging.info(f'Inception Scores: mean={inception_score_mean.detach().item()}, std={inception_score_std.detach().item()}')
        inception.cpu()
        del inception
        logging.info(f'Clip scores: mean={mean(clip_scores)}, std={std(clip_scores)}, max={max(clip_scores)}, min={min(clip_scores)}')
        logging.info(f'All clip scores: {clip_scores}')
        logging.info('*'*20)



    if arg.local_rank in [0, -1]: # all variables on cpu
        # _, order = torch.sort(all_gathered_indices)
        if epoch == 1:
            # all texts are saved
            all_gathered_raw_texts = val_dataloader.dataset.idx2str(torch.arange(len(val_dataloader.dataset))) # back to original order
            save_texts(all_gathered_raw_texts, LM_name, experiment_name, epoch) 
        #     save_original_image(all_gathered_original_images[order][:n_samples], LM_name, experiment_name, epoch, all_in_one, format)
        # if arg.generate:
        #     save_generated_image(all_gathered_generated_images[order][:n_samples], LM_name, experiment_name, epoch, all_in_one, format)
        # save_model(arg, model, optimizer, scheduler, train_epoch_loss, val_epoch_loss, LM_name, experiment_name, epoch)
    logging.info('\n\n')
    torch.distributed.barrier() if torch.distributed.is_initialized() else None # sync
    
    return val_epoch_loss


def to_uint8(img: torch.tensor):
    # img: (bs, 3, h, w), float in range(a, b)
    a, b = torch.amin(img, dim=(1,2,3), keepdim=True).expand_as(img), torch.amax(img, dim=(1,2,3), keepdim=True).expand_as(img)
    new_img = (img-a)/(b-a) * 255
    new_img = new_img.to(torch.uint8)
    return new_img


def clip_pipeline(scores: list, all_images, all_texts: str, batch_idx, compute_every, n_computed, device='cuda'):
    n = len(all_texts)
    bs = 100 # no use
    clip_model, preprocess = clip.load('ViT-B/32', device) 
    all_images = all_images.permute(0, 2,3,1)
    all_images = to_uint8(all_images).detach().cpu().numpy()
    from math import ceil
    n_chunks = ceil(n/bs)
    for i in range(n_chunks):
        start, end = i*bs, (i+1)*bs
        img = all_images[start:end]
        txt = all_texts[start:end]
        image_inputs = torch.stack([preprocess(PIL.Image.fromarray(image)) for image in img]).to(device)
        text_inputs = clip.tokenize(txt).to(device)
        scr, _ = clip_model(image_inputs, text_inputs)
        scores.extend(torch.diag(scr).detach().cpu().numpy().tolist())
    if batch_idx % compute_every == 0:
        logging.info(f'Clip scores: mean={mean(scores)}, std={std(scores)}, max={max(scores)}, min={min(scores)}')



def fid_pipeline(fid, all_original, all_generated, batch_idx, compute_every, n_computed):
    # input images: (n, 3, h, w), torch.FloatTensor with values around 0
    n = len(all_original)
    all_original, all_generated = to_uint8(all_original), to_uint8(all_generated)
    
    fid.update(all_original, real=True)
    fid.update(all_generated, real=False)
    if batch_idx %compute_every == 0:
        fid_score = fid.compute().item()
        logging.info(f'Metrics for {n_computed} samples: @ {datetime.datetime.now()}')
        logging.info(f'FID: {fid_score}')

def inception_pipeline(inception, all_generated, batch_idx, compute_every, n_computed, device='cuda'):
    bs = 100 # no use
    all_generated= to_uint8(all_generated)
    n = len(all_generated)
    from math import ceil
    n_chunks = ceil(n/bs)
    for i in range(n_chunks):
        start, end = i*bs, (i+1)*bs
        g_ = all_generated[start:end].to(device)
        inception.update(g_) # input generated images
    if batch_idx %compute_every == 0:
        inception_score_mean, inception_score_std = inception.compute()
        logging.info(f'Inception Scores: mean={inception_score_mean.detach().item()}, std={inception_score_std.detach().item()}')

        


def save_texts(texts: tuple, LM_name, experiment_name, epoch):
    if epoch != 1:
        return
    check_dir(f'saved/{experiment_name}')
    pth = os.path.join(f'saved/{experiment_name}', f'captions_{LM_name}_{experiment_name}.txt')
    with open(pth, 'w') as f:
        for line in texts:
            f.write(line + '\n')
    print(f'Captions saved to {pth}')

def save_original_image(images, indices, LM_name, experiment_name, epoch, all_in_one=False, format='pdf'):
    # if epoch !=1:
    #     return
    check_dir(f'saved/{experiment_name}')
    if all_in_one:
        ori_img = torchvision.utils.make_grid(images, nrow=8, normalize=True, range=(images.min(), images.max()))
        ori_img_pth = os.path.join(f'saved/{experiment_name}', f'original_{LM_name}_{experiment_name}.{format}')
        plt.imsave(ori_img_pth, ori_img.permute(1,2,0).cpu().numpy(), format=format)
        print(f'Original images are saved to {ori_img_pth}.')
    else:
        dir = f'saved/{experiment_name}/original'
        check_dir(dir)
        # for i, image in enumerate(images,start=1):
        for i, image in zip(indices, images):
            rec_img_pth = os.path.join(dir, f'{i+1}.{format}') # 0-based -> 1-based
            torchvision.utils.save_image(image, rec_img_pth, format, \
                normalize=True, range=(image.min(), image.max()))
        print(f'{len(indices)} original images in {format} format saved to {dir}')


def save_generated_image(generated_images, indices, LM_name, experiment_name, epoch, all_in_one=False, format='pdf'):
    format = 'pdf'
    check_dir(f'saved/{experiment_name}')
    if all_in_one:
        rec_img = torchvision.utils.make_grid(generated_images, \
            nrow=8, normalize=True, range=(generated_images.min(), generated_images.max()))
        rec_img_pth = os.path.join(f'saved/{experiment_name}', f'generated_{LM_name}_{experiment_name}_E{epoch}.{format}')
        plt.imsave(rec_img_pth, rec_img.permute(1,2,0).cpu().numpy(), format=format) # c h w -> h w c
        print(f'Images prompted by {LM_name} at epoch {epoch} saved  in {rec_img_pth}.')
    else:
        dir = f'saved/{experiment_name}/E{epoch}'
        check_dir(dir)
        # for i, image in enumerate(generated_images, start=1):
        for i, image in zip(indices, generated_images):
            rec_img_pth = os.path.join(dir, f'{i+1}.{format}')
            torchvision.utils.save_image(image, rec_img_pth, format, \
                normalize=True, range=(image.min(), image.max()))
        print(f'{len(indices)} images in {format} format prompted by {LM_name} at epoch {epoch} saved  in {dir}.')
            


def save_model(arg, model, optimizer, scheduler, train_epoch_loss, val_epoch_loss, LM_name, experiment_name, epoch):
    
    check_dir(f'saved/{experiment_name}')
    dir = f'saved/{experiment_name}/checkpoints'
    check_dir(dir)
    sd = dict(
        model=model.module.state_dict() if arg.ddp else model.state_dict(),
        optimizer=optimizer, 
        scheduler=scheduler, 
        train_epoch_loss=train_epoch_loss, 
        val_epoch_loss=val_epoch_loss,
        epoch=epoch,
    )
    sd_pth = os.path.join(dir, f'E{epoch}.ckpt')
    torch.save(sd, sd_pth)
    print(f'state_dict saved to {sd_pth}')



if __name__ == '__main__':
    main()
'''
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 main_for_cin_transformer_fast_generate.py --ddp --train --lm xxx --tunelm? --mlp? > test.log 2>&1 &
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 main_for_cin_transformer_fast_generate.py --ddp --generate --mlp > test_.log 2>&1 &
'''
