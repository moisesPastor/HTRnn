# Standard packages
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageOps
import torchvision.transforms.functional as Fv

import numpy as np
from collections import OrderedDict

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# A custom Dataset class must implement three functions: 
# __init__, __len__, and __getitem__. 

class FixedHeightResize(object):
    """
    from https://github.com/pytorch/vision/issues/908
    """
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        size = (self.height, self._calc_new_width(img))
        return Fv.resize(img, size)

    def _calc_new_width(self, img):
        old_width, old_height = img.size
        aspect_ratio = old_width / old_height
        return round(self.height * aspect_ratio)


class HTRDataset(Dataset):
    # Dataset class is for dataset’s features and labels retrieving
    # one sample at a time. 
    
    # A custom Dataset class must implement __init__ function
    def __init__(self, root_dir, spaceChar, fixed_height=64,transform=None, charVoc=None):
        self.root_dir = root_dir
        files = os.listdir(self.root_dir)

        self.fixed_height=fixed_height
        self.items = [fname.rsplit('.',1)[0] for fname in files]        
        self.items =  list(OrderedDict.fromkeys(self.items))

        self.char_voc = charVoc
        self.decoder = dict(enumerate(charVoc))
        self.encoder = dict(zip(self.decoder.values(),self.decoder.keys()))
        self.spaceChar = spaceChar
        
        if charVoc is None:
            self.labels, self.char_voc = self._build_labels_and_char_voc()
        else:
            assert self.spaceChar in charVoc, f'Space symbol \"{self.spaceChar}\" must be included in the character vocabulary of the model!'
            self.labels, _ = self._build_labels_and_char_voc()


    def _build_labels_and_char_voc(self):
        char_voc = set([])
        labels = []

        for item_name in self.items:
            with open(os.path.join(self.root_dir, item_name + ".txt"), 'r') as gt:
                text = gt.read().strip()
                assert not(self.spaceChar in text), f'Chosen space symbol \"{self.spaceChar}\" is already present in the GT vocabulary!'
                label = [c if c!=' ' else self.spaceChar for c in text]
                labels.append(label)
                char_voc |= set(label)
                       
        return np.array(labels, dtype=object), char_voc


    # A custom Dataset class must implement __len__ function
    def __len__(self):
        return len(self.items)

    def get_num_classes(self):
        return len(self.char_voc)

    def get_charVoc(self):
        return self.char_voc

    def get_spaceChar(self):
        return self.spaceChar;

    def get_encoded_label(self, index):
        res=list() 
        for s in self.labels[index]:
            enc = list()
            for c in s:
                pos = self.encoder[c];
                enc.append(pos)

            res.append(enc)               

        return res
    
    def get_decoded_label(self, label):
        return "".join(c if c!=self.spaceChar else ' ' for c in self.char_voc[label])

    # A custom Dataset class must implement __getitem__ function
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_name = self.items[idx]
        
        image = Image.open(os.path.join(self.root_dir, item_name + ".jpg"))
        image = image.convert('L')

        transform = transforms.Compose([transforms.Lambda(lambda x: ImageOps.invert(x)),
                                     FixedHeightResize(height=self.fixed_height),
                                     transforms.ToTensor()])

        image = transform(image)
        
        return image, self.get_encoded_label([idx])[0], idx


# This is required by the DataLoader
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/docs/stable/data.html
def ctc_collate(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    idxs = [item[2] for item in batch]

    #  x ---> list of N Tensor:CxHxW_i 
    # xi ---> CxHxW_i, W_i is the seq length of the ith sample
    input_lengths = [xi.size(2) for xi in x]

    x = zero_pad(x)
    #  x ---> list of N Tensor:CxHxW', 
    #                W' is now the same for all bached samples
    
    #  y ---> list of N of lists of L Ints
    # yi ---> L, L: length of the coded GT line
    target_lengths = [len(yi) for yi in y]
    
    labels = torch.IntTensor(np.hstack(y))
    x = torch.stack(x)
    # x ---> Tensor:NxCxHxW'

    return (x, input_lengths), (labels, target_lengths), idxs


# Reference:
#   N: mini bach size
#   C: number of channels
#   H: height of feature maps
# W_i: width of the ith feature map
#
# For each mini-batch, this function add zeros to all the samples sequences 
# whose lengths were lesser than the sample sequence of maximum length in 
# that mini-batch. So, all the sample sequences will have the same length.
# This is required by the above "ctc_collate" function
def zero_pad(x):
    #  x ---> list of N Tensor:CxHxW_i 
    # xi ---> Tensor:CxHxW_i, W_i is the seq length of the ith sample
    max_w = max(xi.shape[2] for xi in x)
    
    shape = (1, x[0].shape[1], max_w)
    # shape ---> 1xHxW', W' is max_w

    out = []
    for xi in x:
        o = torch.zeros(shape)
        o[:, :, :xi.shape[2]] = xi
        out.append(o)

    # out ---> list of N Tensor:CxHxW'
    return out

