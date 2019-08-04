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
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

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


def split_data(train, test, num_train, num_valid, pretrained_weights, batch_size):
    X_train, y_train = preprocess_dataset(train)
    X_test, y_test = preprocess_dataset(test)

    valid_slice = slice(num_train, num_train + num_valid)
    train_slice = slice(0, num_train)
    X_valid = X_train[valid_slice]
    y_valid = y_train[valid_slice]
    X_train = X_train[train_slice]
    y_train = y_train[train_slice]

    tokenizer = TokenizerTransformer(pretrained_weights)
    X_train, X_valid, X_test = map(tokenizer.fit_transform, (X_train, X_valid, X_test))

    def _batch(X, y, n):
        n_batches = n / batch_size
        X = np.split(X, n_batches)
        y = np.split(y, n_batches)
        return X, y

    X_train, y_train = _batch(X_train, y_train, num_train)
    X_valid, y_valid = _batch(X_valid, y_valid, num_valid)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def _validation_save(model):
    with open(VAL_SAVEPATH, 'wb+') as outfile:
        pickle.dump(model, outfile)


def _validation_load():
    with open(VAL_SAVEPATH, 'rb') as infile:
        model = pickle.load(infile)
    return model


def fit(model, optimizer, criterion, max_epochs, X_train, y_train, X_valid, y_valid, num_valid):
    print('epoch\t\ttrain_loss\tvalid_loss\tvalid_acc\ttime')
    print('=======================================================================')
    best_loss = np.inf
    for epoch in range(max_epochs):
        t0 = time()
        running_loss = 0.0
        for i, batch in enumerate(zip(X_train, y_train), 1):
            optimizer.zero_grad()
            X_i, y_i = batch
            X_i = X_i.to(device)
            y_i = t.LongTensor(y_i)
            y_hat_i = model(X_i)
            loss = criterion(y_hat_i, y_i)
            loss.backward()
            optimizer.step()
            del X_i
            del y_i
            del y_hat_i
            torch.cuda.empty_cache()
            running_loss += loss.item()
        with torch.no_grad():
            num_correct = 0
            valid_loss = 0
            for X_i, y_i in zip(X_valid, y_valid):
                X_i = X_i.to(device)
                y_i = t.LongTensor(y_i)
                y_hat_i = model(X_i)
                valid_loss += criterion(y_hat_i, y_i).item()
                _, predicted = torch.max(y_hat_i.data, 1)
                num_correct += (predicted == y_i).sum().item()
                del X_i
                del y_i
                del y_hat_i
                torch.cuda.empty_cache()
            valid_loss /= len(X_valid)
            running_loss /= len(X_train)
            if valid_loss < best_loss:
                best_loss = valid_loss
                _validation_save(model)
        accuracy = num_correct / num_valid
        dt = time() - t0
        print('{}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}'
              .format(epoch, running_loss, valid_loss, accuracy, dt))
    model = _validation_load()
    return model


def predict(model, X_test):
    preds = []
    with torch.no_grad():
        for X_i in X_test:
            X_i = X_i.to(device)
            y_hat_i = model(X_i)
            _, predicted = torch.max(y_hat_i.data, 1)
            predicted = predicted.cpu()
            preds.append(predicted.numpy())
            del X_i
            del y_hat_i
            torch.cuda.empty_cache()
    return preds


def get_scores(y_test, y_pred_test):
    scores = {}
    y_test, y_pred_test = map(np.concatenate, y_test, y_pred_test)
    scores['f1'] = f1_score(y_test, y_pred_test, average='micro')  # scibert paper uses micro f1 score
    scores['acc'] = accuracy_score(y_test, y_pred_test)
    scores['roc_auc'] = roc_auc_score(y_test, y_pred_test)
    return scores


def save_results(scores, pretrained_weights, num_train, num_valid, batch_size, max_epochs, lr, optimizer, criterion,
                 train_bert):
    results = scores
    results['weights'] = pretrained_weights.split('/')[-1]
    results['num_train'] = num_train
    results['num_valid'] = num_valid
    results['batch_size'] = batch_size
    results['max_epochs'] = max_epochs
    results['optimiser'] = str(optimizer)
    results['learning_rate'] = lr
    results['loss_fn'] = str(criterion)
    results['train_bert'] = train_bert
    timestamp = '{:.0f}'.format(time())
    shutil.copyfile(VAL_SAVEPATH, os.path.join('model_saves', '{}.pickle'.format(timestamp)))
    with open(os.path.join('results', '{}.json'.format(timestamp)), 'w+') as outfile:
        json.dump(results, outfile)


def pretrain(dataset, pretrained_weights, num_train=1024, num_valid=256, batch_size=16, max_epochs=100,
             lr=1e-3, optimizer=optim.Adam, criterion=nn.CrossEntropyLoss, train_bert=True):
    train, test = load_data(dataset)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(train, test, num_train, num_valid,
                                                                    pretrained_weights, batch_size)

    model = BertClassifier(pretrained_weights, y_test.max() + 1, train_bert=train_bert).to(device)
    criterion = criterion(reduction='mean')
    optimizer = optimizer(model.parameters(), lr=lr)
    model = fit(model, optimizer, criterion, max_epochs, X_train, y_train, X_valid, y_valid, num_valid)
    y_pred_test = predict(model, X_test)
    scores = get_scores(y_test, y_pred_test)
    save_results(scores, pretrained_weights, num_train, num_train, batch_size, max_epochs, lr, optimizer, criterion,
                 train_bert)


if __name__ == '__main__':
    pretrain('pubmed20k', os.path.join('bert_weights', 'scibert'), train_bert=False)
