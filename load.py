from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn.functional as F
STEP = 256

def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)

class Preproc:

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        print("self.mean : ", self.mean, " self.std : ", self.std)
        self.classes = sorted(set(l for label in labels for l in label))
        print("self.classes : ", self.classes)
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        print("self.int_to_class : ", self.int_to_class)
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}
        print("self.class_to_int : ", self.class_to_int)
    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        # TODO, awni, fix hack pad with noise for cinc
        print("y in process_y before pad : \n", y)
        y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32) 
        print("y in process_y after pad : \n", y)
        print("y.shape in process_y after pad : ", y.shape)
        # y = keras.utils.np_utils.to_categorical(
        #         y, num_classes=len(self.classes))
        # y = to_categorical(y, num_classes=len(self.classes)) 
        y = torch.tensor(y, dtype=torch.long)
        y = F.one_hot(y, num_classes=len(self.classes))
        print("y in process_y after keras : \n", y)
        print("y.shape in process_y after keras : \n", y.shape)
        return y

def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]



"""Definition of Dataloader"""

class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, ecgs, labels, preproc, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.ecgs = ecgs
        self.labels = labels
        self.prepoc = preproc
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################
        
        num_examples = len(self.ecgs)
        examples = zip(self.ecgs, self.labels)
        examples = sorted(examples, key=lambda x: x[0].shape[0])
        end = num_examples - self.batch_size + 1
        batches = [examples[i:i+self.batch_size] for i in range(0, end, self.batch_size)]
        random.shuffle(batches)
        while True:
            for batch in batches:
                x, y = zip(*batch)
                yield preproc.process(x, y)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last (self.drop_last)!                #
        ########################################################################
        

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length

def build_batches(dataset, batch_size):
    batches = []  # list of all mini-batches
    batch = []  # current mini-batch
    for i in range(len(dataset)):
        batch.append(dataset[i])
        if len(batch) == batch_size:  # if the current mini-batch is full,
            batches.append(batch)  # add it to the list of mini-batches,
            batch = []  # and start a new mini-batch
    return batches


def combine_batch_dicts(batch):
    batch_dict = {}
    for data_dict in batch:
        for key, value in data_dict.items():
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(value)
    return batch_dict

def batch_to_numpy(batch):
    numpy_batch = {}
    for key, value in batch.items():
        numpy_batch[key] = np.array(value)
    return numpy_batch

def build_batch_iterator(dataset, batch_size, shuffle=True):
    if shuffle:
        index_iterator = iter(np.random.permutation(len(dataset)))  # define indices as iterator
    else:
        index_iterator = iter(range(len(dataset)))  # define indices as iterator

    batch = []
    for index in index_iterator:  # iterate over indices using the iterator
        batch.append(dataset[index])
        if len(batch) == batch_size:
            yield batch  # use yield keyword to define a iterable generator
            batch = []











if __name__ == "__main__":
    data_json = "examples/cinc17/train.json"
    train = load_dataset(data_json)
    preproc = Preproc(*train)
    gen = data_generator(32, preproc, *train)
    for x, y in gen:
        print(x.shape, y.shape)
        break