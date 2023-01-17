from math import log2, sqrt
from typing import Iterable

import torch
import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from dalle_pytorch import distributed_utils, DiscreteVAE
from dalle_pytorch.transformer import Transformer, DivideMax
from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE
from einops import rearrange
from torch import nn, einsum

from transformers import BertTokenizer, BertModel
from transformers import CLIPTokenizer, CLIPTextModel


# helpers

def exists(val):
    return val is not None


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class always():
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return self.val


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# sampling helpers

def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs



class Text2ImageModel(nn.Module):
    def __init__(
            self,
            *,
            dim,
            vae,
            num_text_tokens=0,
            text_seq_len=0,
            depth,
            heads=8,
            dim_head=64,
            reversible=False,
            attn_dropout=0.,
            ff_dropout=0,
            t2i=False,
            LM_name=None,
            text_embdim=0,
            sparse_attn=False,
            attn_types=None,
            loss_img_weight=7,
            stable=False,
            sandwich_norm=False,
            shift_tokens=True,
            rotary_emb=True,
            shared_attn_ids=None,
            shared_ff_ids=None,
            share_input_output_emb=False,
            optimize_for_inference=False,
    ):
        super().__init__()
        assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE)), 'vae must be an instance of DiscreteVAE'

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2

        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(
            image_fmap_size, image_fmap_size)) if not rotary_emb else always(0)

        self.num_image_tokens = num_image_tokens

        seq_len = image_seq_len
        self.image_seq_len = image_seq_len

        self.dim = dim
        self.vae = vae

        self.projection_layer = nn.Linear(text_embdim, dim) if t2i else None

        self.transformer = Transformer(
            dim=dim,
            causal=True,
            seq_len=seq_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            reversible=reversible,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            attn_types=attn_types,
            image_fmap_size=image_fmap_size,
            sparse_attn=sparse_attn,
            stable=stable,
            sandwich_norm=sandwich_norm,
            shift_tokens=shift_tokens,
            rotary_emb=rotary_emb,
            shared_attn_ids=shared_attn_ids,
            shared_ff_ids=shared_ff_ids,
            optimize_for_inference=optimize_for_inference,
        )

        self.stable = stable
        self.t2i = t2i
        if LM_name == 'bert':
            self.LM = BertModel.from_pretrained("bert-base-uncased")
            self.LM_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif LM_name == 'clip':
            self.LM = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.LM_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_image_tokens),
        )

        self.image_emb = nn.Embedding(num_image_tokens + 1, dim)
        self.loss_img_weight = loss_img_weight

        set_requires_grad(self.LM, False)
        set_requires_grad(self.vae, False)  # freeze VAE from being trained
        set_requires_grad(self.projection_layer, t2i) if t2i else None
        set_requires_grad(self.image_emb, (not t2i))
        set_requires_grad(self.image_pos_emb, (not t2i)) if not rotary_emb else None
        set_requires_grad(self.transformer, (not t2i))
        set_requires_grad(self.to_logits, (not t2i))

    @torch.no_grad()
    @eval_decorator
    def generate_images(
            self,
            text=None,
            *,
            clip=None,
            filter_thres=0.5,
            temperature=1.,
            img=None,
            num_init_img_tokens=None,
            cond_scale=1.,
            use_cache=False,
    ):
        bs = len(text) if text is not None else 1

        cache = {} if use_cache else None

        if self.t2i: # stage 2
            # serves as a placeholder, will not be used and will be discarded in line 203
            image = torch.empty((bs, 1), dtype=torch.long).to(next(self.parameters()).device) # (bs, 1), long
        else: # stage 1
            image = torch.full((bs, 1), self.num_image_tokens).to(next(self.parameters()).device)

        for cur_len in range(self.image_seq_len):
            logits = self.forward_with_cond_scale(text, image, cond_scale=cond_scale, cache=cache)
            logits = logits[:, -1, :]
            filtered_logits = top_k(logits, thres=filter_thres)
            sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            image = torch.cat((image, sample[:, None]), dim=-1)

        images = self.vae.decode(image[:, 1:])

        if exists(clip):
            scores = clip(text, images, return_loss=False)
            return images, scores

        return images

    def forward_with_cond_scale(self, *args, cond_scale=1, cache=None, **kwargs):
        if cond_scale == 1:
            return self(*args, **kwargs)

        prev_cache = cache.copy() if exists(cache) else None
        logits = self(*args, cache=cache, **kwargs)

        # discovery by Katherine Crowson
        # https://twitter.com/RiversHaveWings/status/1478093658716966912
        null_cond_logits = self(*args, null_cond_prob=1., cache=prev_cache, **kwargs)
        return null_cond_logits + (logits - null_cond_logits) * cond_scale

    def forward(
            self,
            text=None,
            image=None,
            return_loss=False,
            null_cond_prob=0.,
            cache=None,
    ):

        if len(image.shape) == 4: # training
            image_size = self.vae.image_size
            channels = self.vae.channels
            assert tuple(image.shape[1:]) == (channels, image_size, image_size)
            image = self.vae.get_codebook_indices(image)
            labels = image.clone()
            if self.t2i: # stage 2, get corresponding embeddings then concat
                text = self.get_embedding(text, self.LM, self.LM_tokenizer, image.device)
                projected_text_emb = self.projection_layer(text)
                bos = projected_text_emb # (bs, 1, dim)
                image_emb = self.image_emb(image[:, :-1]) # (bs, img_seqlen-1, dim)
                tokens = torch.cat([bos, image_emb], dim=1) # (bs, img_seqlen, dim)

            else: # stage 1, concat then get embedding
                bos = torch.full((image.size(0), 1), self.num_image_tokens).to(image) # <bos> corresponds to index 8192
                image = torch.cat([bos, image[:, :-1]], dim=1)
                tokens = self.image_emb(image) # (bs, img_seqlen, ebddim)

        else: # invoked in generate_image()
            if self.t2i: # stage 2
                _, cur_len = image.shape # (bs, current_length)
                text = self.get_embedding(text, self.LM, self.LM_tokenizer, image.device)
                tokens = self.projection_layer(text) # (bs, 1, dim)
                if cur_len > 1: # has image token
                    image_emb = self.image_emb(image[:, 1:]) # (bs, curlen-1, dim)
                    tokens = torch.cat([tokens, image_emb], dim=1) # (bs, curlen, dim)
            else: # stage 1
                tokens = self.image_emb(image)

        tokens += self.image_pos_emb(tokens) # no effect


        if self.stable:
            alpha = 0.1
            tokens = tokens * alpha + tokens.detach() * (1 - alpha)

        if exists(cache) and cache.get('offset'):
            tokens = tokens[:, -1:]
        out = self.transformer(tokens, cache=cache)

        if self.stable: # default False
            out = self.norm_by_max(out)

        logits = self.to_logits(out)

        if exists(cache):
            cache['offset'] = cache.get('offset', 0) + logits.shape[1]

        if not return_loss:
            return logits

        assert exists(image), 'when training, image must be supplied'

        logits = rearrange(logits, 'b n c -> b c n')
        loss = F.cross_entropy(logits, labels)
        return loss

    @torch.no_grad()
    def get_embedding(self, text, model, tokenizer, device):
        if not isinstance(text, Iterable):
            text = [text]
        inputs = tokenizer(text, padding=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        pooled_output = outputs.pooler_output  # pooled (EOS token) states
        return pooled_output.unsqueeze(1) # (bs, 1, dim), (768 for bert; 512 for clip)
