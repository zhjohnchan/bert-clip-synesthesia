from math import log2, sqrt

import torch
import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from dalle_pytorch import distributed_utils, DiscreteVAE
from dalle_pytorch.transformer import Transformer, DivideMax
from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE
from einops import rearrange
from torch import nn, einsum


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

        self.vae = vae
        set_requires_grad(self.vae, False)  # freeze VAE from being trained

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

        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_image_tokens),
        )

        self.image_emb = nn.Embedding(num_image_tokens + 1, dim)
        self.loss_img_weight = loss_img_weight

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

        if len(image.shape) == 4:
            image_size = self.vae.image_size
            channels = self.vae.channels
            assert tuple(image.shape[1:]) == (channels, image_size, image_size)
            image = self.vae.get_codebook_indices(image)
            labels = image.clone()
            bos = torch.full((image.size(0), 1), self.num_image_tokens).to(image)
            image = torch.cat([bos, image[:, :-1]], dim=1)

        image_emb = self.image_emb(image)
        image_emb += self.image_pos_emb(image_emb)

        # TODO: add text prefix embeddings here when linear probing
        tokens = image_emb

        if self.stable:
            alpha = 0.1
            tokens = tokens * alpha + tokens.detach() * (1 - alpha)

        if exists(cache) and cache.get('offset'):
            tokens = tokens[:, -1:]
        out = self.transformer(tokens, cache=cache)

        if self.stable:
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
