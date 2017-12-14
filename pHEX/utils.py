import time
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from sklearn.model_selection import StratifiedShuffleSplit


class DogsDataset(Dataset):
    def __init__(self, labels, root_dir, subset=False, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname)
        labels = self.labels.iloc[idx, 1:].as_matrix().astype('float')
        labels = np.argmax(labels)
        if self.transform:
            image = self.transform(image)
        return [image, labels]

def labels_to_pivot(labels):

    labels['target'] = 1
    labels['rank'] = labels.groupby('breed').rank()['id']
    labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
    return labels_pivot


def train_valid_split(X, y, labels):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 123)
    train_index, val_index = next(sss.split(X, y))
    train_labels = labels.loc[train_index,:]
    val_labels = labels.loc[val_index,:]
    train = labels_to_pivot(train_labels)
    valid = labels_to_pivot(val_labels)
    return train, valid


def data_loader(train, valid, args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ds_trans = transforms.Compose([transforms.Scale(args.input_size),
                                   transforms.CenterCrop(args.input_size),
                                   transforms.ToTensor(),
                                   normalize])
    train_ds = DogsDataset(train, args.data_dir + 'train/', transform=ds_trans)
    valid_ds = DogsDataset(valid, args.data_dir + 'train/', transform=ds_trans)
    # test_ds = DogsDataset(sample_submission, args.data_dir + 'test/', transform=ds_trans)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    # test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=True, num_workers=4)

    return train_dl, valid_dl


def evaluate_model(model, batch,criterion, args):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0

    model.eval()

    for inputs, labels in batch:

        if args.cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        total += labels.size(0)
        correct += torch.sum(preds == labels.data)


    model.train()

    return loss.data[0], (100 * correct / total)
