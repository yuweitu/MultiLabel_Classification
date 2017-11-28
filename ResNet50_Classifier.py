import time
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import argparse
import random
import torchMobileNet
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

from dataloader import DogsDataset


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
    #sample_submission = pd.read_csv(join(args.data_dir, 'sample_submission.csv'))

    selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(args.num_classes).index)
    labels = labels[labels['breed'].isin(selected_breed_list)]
    labels['target'] = 1
    labels['rank'] = labels.groupby('breed').rank()['id']
    labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)

    train = labels_pivot.sample(frac=0.8)
    valid = labels_pivot[~labels_pivot['id'].isin(train['id'])]
    print(train.shape, valid.shape)

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
    #test_ds = DogsDataset(sample_submission, args.data_dir + 'test/', transform=ds_trans)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    #test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=True, num_workers=4)

    resnet = models.resnet50()

    # freeze all model parameters
    for param in resnet.parameters():
        param.requires_grad = True

    # new final layer with 16 classes
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, args.num_classes)

    if args.cuda:
        resnet = resnet.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.fc.parameters(), lr = args.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    start_time = time.time()
    model = train_model(args,train_dl, valid_dl, len(train), resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs)
    torch.save(model, 'dog_ResNet.pt')
    #out = model()
    #test_loss, test_acc = evaluate_model(model, test_dl, criterion, args)
    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
    #print('Test loss : {:10f}; Test Accuracy: {:10f}'.format(test_loss, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', type=int, default=10)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-num_classes', type=float, default=16)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-data_dir', type=str, default = 'input/')
    args = parser.parse_args()
    args.cuda = False

    main(args)