"""
Module to pretrain sentence classifier
"""
import os
import shutil
from tempfile import gettempdir
import json
from time import time
import pickle
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from model import TokenizerTransformer, BertClassifier

VAL_SAVEPATH = os.path.join(gettempdir(), 'model.pickle')

use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
t = torch.cuda if use_cuda else torch


def load_data(dataset):
    dataset_dir = os.path.join('datasets', dataset)
    train = pd.read_csv(os.path.join(dataset_dir, 'train.csv'), index_col=0)
    test = pd.read_csv(os.path.join(dataset_dir, 'test.csv'), index_col=0)
    return train, test


def preprocess_dataset(ds):
    X = ds['sentence'].values
    y = ds['label'].values
    return X, y


class BertTokenDataset(Dataset):

    def __init__(self, X, y, pretrained_weights):
        tokenizer = TokenizerTransformer(pretrained_weights)
        X = tokenizer.fit_transform(X)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        X_i = self.X[i]
        y_i = self.y[i]
        return X_i, y_i


class AbstractDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class BertTokenPlusDictDataset(BertTokenDataset):

    def __init__(self, X, y, pretrained_weights, count_vectorizer):
        super().__init__(X, y, pretrained_weights)
        self.X_count = count_vectorizer.fit_transform(X).todense()

    def __getitem__(self, i):
        X_bert_i, y_i = super().__getitem__(i)
        X_count_i = self.X_count[i, :]
        X_i = X_bert_i, X_count_i
        return X_i, y_i


def split_data(train, test, num_train, num_valid, num_test, pretrained_weights, batch_size, model='bert_only'):
    X_train, y_train = preprocess_dataset(train)
    X_test, y_test = preprocess_dataset(test)

    valid_slice = slice(num_train, num_train + num_valid)
    train_slice = slice(0, num_train)
    X_valid = X_train[valid_slice]
    y_valid = y_train[valid_slice]
    X_train = X_train[train_slice]
    y_train = y_train[train_slice]
    X_test = X_test[:num_test]
    y_test = y_test[:num_test]

    trainloader = DataLoader(AbstractDataset(X_train, y_train), batch_size=batch_size)
    validloader = DataLoader(AbstractDataset(X_valid, y_valid), batch_size=batch_size)
    testloader = DataLoader(AbstractDataset(X_test, y_test), batch_size=batch_size)

    return trainloader, validloader, testloader


def _validation_save(model):
    with open(VAL_SAVEPATH, 'wb+') as outfile:
        pickle.dump(model, outfile)


def _validation_load():
    with open(VAL_SAVEPATH, 'rb') as infile:
        model = pickle.load(infile)
    return model


def fit(model, optimizer, criterion, max_epochs, trainloader, validloader):
    print('epoch\t\ttrain_loss\tvalid_loss\tvalid_acc\ttime')
    print('=======================================================================')
    best_loss = np.inf
    for epoch in range(max_epochs):
        t0 = time()
        running_loss = 0.0
        for i, batch in enumerate(trainloader, 1):
            optimizer.zero_grad()
            X_i, y_i = batch
            # X_i = X_i.to(device)
            y_i = y_i.to(device)
            y_hat_i = model(X_i)
            loss = criterion(y_hat_i, y_i)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            del X_i
            del y_i
            del y_hat_i
            del loss
            torch.cuda.empty_cache()
        with torch.no_grad():
            num_correct = 0
            valid_loss = 0
            for j, (X_j, y_j) in enumerate(validloader, 1):
                # X_j = X_j.to(device)
                y_j = y_j.to(device)
                y_hat_j = model(X_j)
                loss = criterion(y_hat_j, y_j)
                valid_loss += loss.item()
                _, predicted = torch.max(y_hat_j.data, 1)
                num_correct += (predicted == y_j).sum().item()
                del X_j
                del y_j
                del y_hat_j
                del loss
                del predicted
                torch.cuda.empty_cache()
            valid_loss /= j
            running_loss /= i
            if valid_loss < best_loss:
                best_loss = valid_loss
                _validation_save(model)
        accuracy = num_correct / len(validloader.dataset)
        dt = time() - t0
        print('{}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}'
              .format(epoch, running_loss, valid_loss, accuracy, dt))
    model = _validation_load()
    return model


def predict(model, testloader):
    preds = []
    with torch.no_grad():
        for X_i, y_i in testloader:
            # X_i = X_i.to(device)
            y_hat_i = model(X_i)
            _, predicted = torch.max(y_hat_i.data, 1)
            predicted = predicted.cpu()
            preds.append(predicted.numpy())
            del X_i
            del y_hat_i
            del predicted
            torch.cuda.empty_cache()
    return preds


def get_scores(y_test, y_pred_test):
    scores = {}
    y_pred_test = np.concatenate(y_pred_test)
    scores['f1_score'] = f1_score(y_test, y_pred_test, average='micro')  # scibert paper uses micro f1 score
    scores['accuracy'] = accuracy_score(y_test, y_pred_test)
    return scores


def save_results(save_dir, scores, pretrained_weights, num_train, num_valid, num_test, batch_size, max_epochs, lr,
                 optimizer, criterion, train_bert):
    results = scores
    results['weights'] = pretrained_weights.split('/')[-1]
    results['num_train'] = num_train
    results['num_valid'] = num_valid
    results['num_test'] = num_test
    results['batch_size'] = batch_size
    results['max_epochs'] = max_epochs
    results['optimiser'] = str(optimizer)
    results['learning_rate'] = lr
    results['loss_fn'] = str(criterion)
    results['train_bert'] = train_bert
    timestamp = '{:.0f}'.format(time())

    model_saves_dir = os.path.join(save_dir, 'model_saves')
    results_save_dir = os.path.join(save_dir, 'results')
    for directory in [save_dir, results_save_dir, model_saves_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    shutil.copyfile(VAL_SAVEPATH, os.path.join(model_saves_dir, '{}.pickle'.format(timestamp)))
    with open(os.path.join(results_save_dir, '{}.json'.format(timestamp)), 'w+') as outfile:
        json.dump(results, outfile)


def pretrain(dataset, pretrained_weights, save_dir='.', num_train=1024, num_valid=256, num_test=256, batch_size=16,
             max_epochs=100, lr=1e-3, optimizer=optim.Adam, criterion=nn.CrossEntropyLoss, train_bert=True):
    train, test = load_data(dataset)
    output_dim = test['label'].max() + 1
    trainloader, validloader, testloader = split_data(train, test, num_train, num_valid,
                                                      num_test, pretrained_weights, batch_size)

    model = BertClassifier(pretrained_weights, output_dim, train_bert=train_bert).to(device)
    criterion = criterion(reduction='mean')
    optimizer = optimizer(model.parameters(), lr=lr)
    model = fit(model, optimizer, criterion, max_epochs, trainloader, validloader)
    y_pred_test = predict(model, testloader)
    scores = get_scores(testloader.dataset.y, y_pred_test)
    save_results(save_dir, scores, pretrained_weights, num_train, num_train, num_test, batch_size, max_epochs, lr,
                 optimizer, criterion, train_bert)


if __name__ == '__main__':
    pretrain('pubmed20k', os.path.join('bert_weights', 'scibert'), train_bert=False, num_train=17, num_valid=17,
             num_test=17, max_epochs=2)
