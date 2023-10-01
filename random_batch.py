import random
import torch
import numpy as np


def random_batch(x_t, y_t, batch_size):
    '''
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(x_t)
    random.seed(randnum)
    random.shuffle(y_t)
    '''
    batch = []
    batch_label = []
    for i in range(0, int(len(x_t) / batch_size)):
        left = 0 + i * batch_size
        right = batch_size + i * batch_size
        cut_x = x_t[left:right]
        cut_y = y_t[left:right]
        batch_i = torch.zeros([batch_size, 768])
        label_i = torch.zeros([batch_size])
        for j in range(0, len(batch_i)):
            cut_x[j] = cut_x[j][None, :]
            batch_i[j, :] = cut_x[j]
            label_i[j] = torch.tensor(cut_y[j])
        batch.append(batch_i)
        batch_label.append(label_i)
    return batch, batch_label
