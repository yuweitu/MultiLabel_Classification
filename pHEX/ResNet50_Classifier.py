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
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from sklearn.model_selection import StratifiedShuffleSplit


from dataloader import DogsDataset,labels_to_pivot,train_valid_split,data_loader,evaluate_model


def train_model(args, X, y, labels, model, criterion, optimizer, scheduler, num_epochs):
    train, valid = train_valid_split(X, y, labels)
    train_size = len(train)
    valid_size = len(valid)
    train_batch, valid_batch = data_loader(train, valid, args)

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
        train_epoch_acc = running_corrects / running_total*100

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

    selected_breed_list = list(
        labels.groupby('breed').count().sort_values(by='id', ascending=False).head(args.num_classes).index)
    labels = labels[labels['breed'].isin(selected_breed_list)].reset_index()

    breed = set(labels['breed'])
    class_to_num = dict(zip(breed, range(args.num_classes)))
    X = np.zeros((len(labels), args.input_size, args.input_size, 3), dtype=np.uint8)
    y = np.zeros(len(labels), dtype=np.uint8)
    for i in range(len(labels)):
        X[i] = cv2.resize(cv2.imread('input/train/%s.jpg' % labels['id'][i]), (args.input_size, args.input_size))
        y[i] = class_to_num[labels['breed'][i]]

    train, valid = train_valid_split(X, y, labels)
    train_size = len(train)
    valid_size = len(valid)
    train_batch, valid_batch = data_loader(train, valid, args)

    #resnet = models.resnet18(pretrained=True)
    resnet = models.resnet182(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False
    num_features = resnet.fc.in_features
    fc_layers = nn.Sequential(
        nn.Linear(num_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(4096, 120),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(4096,args.num_classes)
    )
    resnet.fc = fc_layers


    # new final layer with 120 classes
    #num_ftrs = resnet.fc.in_features
    #resnet.fc = torch.nn.Linear(num_ftrs, args.num_classes)

    if args.cuda:
        resnet = resnet.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.fc.parameters(), lr = args.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    start_time = time.time()
    model = train_model(args, X, y, labels, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs)
    torch.save(model, 'dog_ResNet.pt')
    #out = model()
    #test_loss, test_acc = evaluate_model(model, test_dl, criterion, args)
    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
    #print('Test loss : {:10f}; Test Accuracy: {:10f}'.format(test_loss, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', type=int, default=16)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default=0.05)
    parser.add_argument('-num_classes', type=float, default=16)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-data_dir', type=str, default = 'input/')
    parser.add_argument('-relabel', type=str, default='False')
    args = parser.parse_args()
    args.cuda = False
    args.relabel = True

    main(args)