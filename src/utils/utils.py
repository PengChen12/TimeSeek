
from torch import nn
import collections
from collections import OrderedDict
import torch
import os
from datetime import timedelta
import requests
from tqdm import tqdm
import os


def init_ddp():
    local_rank = int(os.environ.get('LOCAL_RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))
    rank = int(os.environ.get('RANK'))

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=600)
    )


def nested_children(m: nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
                
    return output


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unwrap_model(model):
    unwrapped_model = nested_children(model)
    unwrapped_model = flatten_dict(unwrapped_model)
    unwrapped_model = nn.Sequential(OrderedDict(unwrapped_model))
    return unwrapped_model
    

def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(local_filename, 'wb') as file:
        with tqdm(
            desc=f"Download: {local_filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))