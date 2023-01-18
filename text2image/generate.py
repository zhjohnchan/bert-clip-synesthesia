import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from einops import repeat
from omegaconf import OmegaConf
from transformers import top_k_top_p_filtering

from cond_transformer import Net2NetTransformer

rescale = lambda x: (x + 1.) / 2.


def chw_to_pillow(x):
    return Image.fromarray((255 * rescale(x.detach().cpu().numpy().transpose(1, 2, 0))).clip(0, 255).astype(np.uint8))


@torch.no_grad()
def sample_with_past(text, x, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None):
    sample = x
    cond_len = x.shape[1]
    if text is not None:
        embeddings = model.get_text_embeddings(text)
        embeddings_len = embeddings.shape[1]
    else:
        embeddings_len = 0

    past = None
    for n in range(embeddings_len + steps):
        if callback is not None:
            callback(n)
        if n < embeddings_len:
            _, _, present = model.transformer.forward_with_past(
                embeddings=embeddings[:, n:n + 1], past=past, past_length=(n + cond_len - 1))
            if past is None:
                past = [present]
            else:
                past.append(present)
        else:
            logits, _, present = model.transformer.forward_with_past(
                idx=x, past=past, past_length=(n + cond_len - 1 - embeddings_len))
            if past is None:
                past = [present]
            else:
                past.append(present)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)
            if not sample_logits:
                _, x = torch.topk(probs, k=1, dim=-1)
            else:
                x = torch.multinomial(probs, num_samples=1)
            # append to the sequence and continue
            sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, cond_len:]  # cut conditioning off
    return sample


@torch.no_grad()
def sample_unconditional(text, model, batch_size, steps=256, temperature=None, top_k=None, top_p=None, callback=None,
                         dim_z=256, h=16, w=16, verbose_time=False):
    if isinstance(text, str):
        text = [text] * batch_size
    elif isinstance(text, list):
        assert len(text) == batch_size
    else:
        print("No text prompts.")

    log = dict()
    qzshape = [batch_size, dim_z, h, w]
    assert model.be_unconditional, 'Expecting an unconditional model.'
    c_indices = repeat(torch.tensor([model.sos_token]), '1 -> b 1', b=batch_size).to(model.device)  # sos token
    t1 = time.time()
    index_sample = sample_with_past(text=text, x=c_indices, model=model, steps=steps,
                                    sample_logits=True, top_k=top_k, callback=callback,
                                    temperature=temperature, top_p=top_p)
    if verbose_time:
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")
    x_sample = model.decode_to_img(index_sample, qzshape)
    log["samples"] = x_sample
    return log


def save_images(images, save_dir, filenames):
    root = os.path.join(save_dir, "images")
    os.makedirs(root, exist_ok=True)
    for image, filename in zip(images, filenames):
        x = chw_to_pillow(image)
        path = os.path.join(root, filename)
        x.save(path)


def load_model(text_model_name, ckpt_path, learning_rate=0.0625):
    config_path = "2021-04-23T18-11-19_celebahq_transformer/configs/2021-04-23T18-11-19-project.yaml"
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    config = OmegaConf.load(config_path)
    model = Net2NetTransformer(**config.model.params, learning_rate=learning_rate, text_model_name=text_model_name)
    missing, unexpected = model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda()
    model.eval()
    return model


@torch.no_grad()
def main():
    top_k = 250
    top_p = 1.0
    batch_size = 1
    temperature = 1.0

    # un-tuned
    # text_model_name = "bert-large-uncased"
    # ckpt_path = "2021-04-23T18-11-19_celebahq_transformer/checkpoints/last.ckpt"

    # bert-large-uncased
    text_model_name = "bert-large-uncased"
    ckpt_path = "outputs/bert-large-uncased/last.ckpt"

    save_dir = "generated_images/" + ckpt_path.replace("/", "_")

    model = load_model(text_model_name, ckpt_path)

    for i in range(10):
        log = sample_unconditional("The person has high cheekbones, and pointy nose. She is wearing lipstick.",
                                   model=model, batch_size=batch_size, temperature=temperature,
                                   top_k=top_k, top_p=top_p)
        save_images(log["samples"], save_dir=save_dir, filenames=[f"{i}.png"])


if __name__ == '__main__':
    main()
