import os

import cv2
import numpy as np
import torch

def check_dir(base):
    if os.path.isdir(base):
        pass
    else:
        os.makedirs(base)

def save_state_dir(network, save_model_dir):
    state_dict = network.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, save_model_dir)


def load_state_dir(network, ckpts, device):
    network.load_state_dict({k.replace('module.', ''): v for k, v in ckpts.items()})
    network.to(device)
    network.eval()


def save_img(x, save_dir):
    x = x.squeeze(dim=0).permute(1, 2, 0)
    x = x.cpu().detach().numpy()
    x = (x * 255.).astype(np.float64)
    cv2.imwrite(save_dir, x)
