import time
import numpy as np
import pandas as pd
import cv2
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import argparse
import random
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from sklearn.model_selection import StratifiedShuffleSplit


from dataloader import DogsDataset


def labels_to_pivot(labels):

    labels['target'] = 1
    labels['rank'] = labels.groupby('breed').rank()['id']
    labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
    return labels_pivot


def train_valid_split(X, y, labels):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
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


def train_model(args, train_batch, valid_batch, train_size, model, criterion, optimizer, scheduler, num_epochs):

    best_model_wts = model.state_dict()
    best_acc = 0.000

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        start = 0

        for inputs, labels in train_batch:

            scheduler.step()
            start += 1
            if args.cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            if args.batch*start % 100 == 0:
                print('training epoch {:s}: completed {:3f}%, current loss {:.3f}'
                        .format(str(epoch+1), round(100 * args.batch*start/ train_size,3), loss.data[0]))

            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
            running_total += labels.size()[0]

        train_epoch_loss = running_loss / running_total
        train_epoch_acc = running_corrects / running_total

        valid_epoch_loss, valid_epoch_acc = evaluate_model(model, valid_batch, criterion, args)

        if valid_epoch_acc > best_acc:
            best_acc = valid_epoch_acc
            best_model_wts = model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(epoch+1, num_epochs,train_epoch_loss, train_epoch_acc,
                                                      valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model


def main(args):

    labels = pd.read_csv(join(args.data_dir, 'labels.csv'))
    sample_submission = pd.read_csv(join(args.data_dir, 'sample_submission.csv'))
    print(len(listdir(join(args.data_dir, 'train'))), len(labels))
    print(len(listdir(join(args.data_dir, 'test'))), len(sample_submission))

    breed = set(labels['breed'])
    class_to_num = dict(zip(breed, range(args.num_classes)))
    X = np.zeros((len(labels), args.input_size, args.input_size, 3), dtype=np.uint8)
    y = np.zeros(len(labels), dtype=np.uint8)
    for i in range(len(labels)):
        X[i] = cv2.resize(cv2.imread('input/train/%s.jpg' % labels['id'][i]), (args.input_size, args.input_size))
        y[i] = class_to_num[labels['breed'][i]]

    train, valid = train_valid_split(X, y, labels)
    print(train.shape, valid.shape)

    train_dl, valid_dl = data_loader(train, valid, args)
    resnet = models.resnet50()

    # unfreeze all model parameters
    for param in resnet.parameters():
        param.requires_grad = True

    # new final layer with 120 classes
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, args.num_classes)

    if args.cuda:
        resnet = resnet.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(resnet.fc.parameters(), lr = args.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    start_time = time.time()
    model = train_model(args, train_dl, valid_dl, len(train), resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs)
    torch.save(model, 'dog_ResNet.pt')
    #out = model()
    #test_loss, test_acc = evaluate_model(model, test_dl, criterion, args)
    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
    #print('Test loss : {:10f}; Test Accuracy: {:10f}'.format(test_loss, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', type=int, default=16)
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-num_classes', type=float, default=120)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-data_dir', type=str, default = 'input/')
    args = parser.parse_args()
    args.cuda = True

    main(args)