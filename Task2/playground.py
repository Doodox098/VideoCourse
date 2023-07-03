import torch
import os
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from tqdm.notebook import tqdm

from pathlib import Path

test_image = np.array(Image.open('C:/Users/doodo/Desktop/ИИТ/Task2/tiny_data/test/interlace/0/00000.png'),
                      dtype=np.float32) / 255.
test_image = np.array([test_image])
plt.imshow(test_image[0])
x = torch.tensor(test_image)

def find_motion_vector(block, domain):
    size_of_blocks = (block.size(dim=1), block.size(dim=2))
    min = np.inf
    min_v = torch.rand(block.size())

    def metrics(cmp_block):
        diff = block[0] - cmp_block[0]
        return torch.mean(diff ** 2)

    for i in range(domain.size(dim=1) - block.size(dim=1)):
        for j in range(domain.size(dim=2) - block.size(dim=2)):
            #print(f"domain {i * size_of_blocks[0]}:{(i + 1) * size_of_blocks[0]}, {j * size_of_blocks[1]}:{(j + 1) * size_of_blocks[1]} {domain[i * size_of_blocks[0]:(i + 1) * size_of_blocks[0], j * size_of_blocks[1]:(j + 1) * size_of_blocks[1]].size()}")
            cur_metric = metrics(domain[:, i : i + size_of_blocks[0],
                                 j : j + size_of_blocks[1]])
            if cur_metric < min:
                min = cur_metric
                min_v = domain[:, i : i + size_of_blocks[0],
                                 j : j + size_of_blocks[1]]
    return min_v


size_of_blocks = (16, 16)
size_of_domain = (32, 32)

x = x.permute(0, 3, 1, 2)
cur_even = torch.from_numpy(np.zeros_like(x))
next_odd = torch.from_numpy(np.zeros_like(x))
cur_even[:, :, 1::2] = x[:, :, 1::2]
next_odd[:, :, ::2] = x[:, :, ::2]

print(x.size())

batch_size = x.size(dim=0)
height = x.size(dim=2)
width = x.size(dim=3)
height_halves = height // 2
mv_height = height_halves // size_of_blocks[0]
mv_width = width // size_of_blocks[1]

print(next_odd.size())
print(mv_height)
print(mv_width)
cnt = 0
with torch.no_grad():
    cur = x[:, :, ::2]
    next = x[:, :, 1::2]
    plt.subplot(1, 3, 1), plt.imshow(cur[0].permute(1, 2, 0))
    plt.subplot(1, 3, 2), plt.imshow(next[0].permute(1, 2, 0))
    plt.subplot(1, 3, 3), plt.imshow(next_odd[0].permute(1, 2, 0))
    for batch in range(batch_size):
        for i in tqdm(range(mv_height)):
            for j in range(mv_width):
                block = cur[batch, :, i * size_of_blocks[0]:(i + 1) * size_of_blocks[0],
                        j * size_of_blocks[1]:(j + 1) * size_of_blocks[1]]
                #plt.imshow(block.permute(1, 2, 0))
                domain = next[batch,
                         :,
                         max(0, i * size_of_blocks[0] + size_of_blocks[0]//2 - size_of_domain[0]):
                         min(height_halves, (i + 1) * size_of_blocks[0] - size_of_blocks[0]//2 + size_of_domain[0]),
                         max(0, j * size_of_blocks[1] + size_of_blocks[1]//2 - size_of_domain[1]):
                         min(width, (j + 1) * size_of_blocks[1] - size_of_blocks[1]//2 + size_of_domain[1])
                         ]
                try:
                    next_odd[batch, :, 2 * i * size_of_blocks[0] + 1 : 2 * (i + 1) * size_of_blocks[0] + 1 : 2,
                    j * size_of_blocks[1]: (j + 1) * size_of_blocks[1]] = find_motion_vector(block, domain)

                except:
                    print(i, j)
                    print(next_odd[batch, :, 2 * i * size_of_blocks[0] + 1 : 2 * (i + 1) * size_of_blocks[0] + 1 : 2,
                    j * size_of_blocks[1]: (j + 1) * size_of_blocks[1]].size())
                    print(find_motion_vector(block, domain).size())
                    raise Exception("Fuck")
print(cnt)
