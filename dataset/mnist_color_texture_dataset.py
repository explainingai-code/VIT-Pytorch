import numpy as np
import cv2
import os
import torch
import json
from torch.utils.data.dataset import Dataset

def get_random_crop(image, crop_h, crop_w):
    h, w = image.shape[:2]
    max_x = w - crop_w
    max_y = h - crop_h
    
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_h, x: x + crop_w, :]
    return crop


def get_center_crop(image):
    h, w = image.shape[:2]
    if h > w:
        return image[(h - w) // 2:-(h - w) // 2, :, :]
    else:
        return image[:, (w - h) // 2:-(w - h) // 2, :]


class MnistDataset(Dataset):
    r"""
    Minimal image dataset where we take mnist images
    add a texture background
    change the color of the digit.
    Model trained on this dataset is then required to predict the below 3 values
    1. Class of texture
    2. Class of number
    3. R, G, B values (0-1) of the digit color
    """
    def __init__(self, split, config, im_h=224, im_w=224):
        self.split = split
        self.db_root = config['root_dir']
        self.im_h = im_h
        self.im_w = im_w
        
        imdb = json.load(open(os.path.join(self.db_root,  'imdb.json')))
        self.im_info = imdb['{}_data'.format(split)]
        self.texture_to_idx = imdb['texture_classes_index']
        self.idx_to_texture = {v:k for k,v in self.texture_to_idx.items()}
        
    def __len__(self):
        return len(self.im_info)
    
    def __getitem__(self, index):
        entry = self.im_info[index]
        digit_cls = int(entry['digit_name'])
        digit_im = cv2.imread(os.path.join(self.db_root, entry['digit_image']))
        digit_im = cv2.cvtColor(digit_im, cv2.COLOR_BGR2RGB)
        digit_im = cv2.resize(digit_im, (self.im_h, self.im_w))
        
        # Discretize mnist images to be either 0 or 1
        digit_im[digit_im > 50] = 255
        digit_im[digit_im <= 50] = 0
        mask_val = (digit_im > 0).astype(np.float32)
        digit_im = np.concatenate((digit_im[:, :, 0][..., None] * float(entry['color_r']),
                                   digit_im[:, :, 1][..., None] * float(entry['color_g']),
                                   digit_im[:, :, 2][..., None] * float(entry['color_b'])), axis=-1)
        im = cv2.imread(os.path.join(self.db_root, entry['texture_image']))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.split == 'train':
            im = get_random_crop(im, self.im_h, self.im_w)
        else:
            im = get_center_crop(im)
            im = cv2.resize(im, (self.im_h, self.im_w))
        out_im = mask_val * digit_im + (1 - mask_val) * im
        im_tensor = torch.from_numpy(out_im).permute((2, 0, 1))
        im_tensor = 2 * (im_tensor / 255) - 1
        return {
            "image" : im_tensor,
            "texture_cls" : self.texture_to_idx[entry['texture_name']],
            "number_cls" : digit_cls,
            "color":torch.as_tensor([float(entry['color_r']),
                                     float(entry['color_g']),
                                      float(entry['color_b'])])
        }
