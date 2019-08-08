"""
Module to pretrain sentence classifier
"""
import os
import shutil
from tempfile import gettempdir
import json
from time import time
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from pubmed_text_classification.model import TransitonModel
from pubmed_text_classification.datasets import PubMed20kPrevious

model_cls, dataset_cls = TransitonModel, PubMed20kPrevious

VAL_SAVEPATH = os.path.join(gettempdir(), 'model.pickle')

use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
t = torch.cuda if use_cuda else torch
print('Running on {}'.format(device))


def _validation_save(model):
    torch.save(model.state_dict(), VAL_SAVEPATH)


def _validation_load(pretrained_weights, train_embeddings, **model_params):
    model = model_cls(pretrained_weights, 5, train_embeddings=train_embeddings, **model_params)
    model.load_state_dict(torch.load(VAL_SAVEPATH))
    return model.to(device)


def _get_datasets(num_train, num_test, valid_split):
    trainset = dataset_cls('train', num_load=num_train)
    testset = dataset_cls('test', num_load=num_test)
    num_valid = int(valid_split*len(trainset))
    num_train = len(trainset) - num_valid
    trainset, validset = random_split(trainset, [num_train, num_valid])
    return trainset, validset, testset


def fit(model, optimizer, criterion, max_epochs, trainloader, validloader, pretrained_weights, train_embeddings,
        **model_params):
    print('epoch\t\ttrain_loss\tvalid_loss\tvalid_acc\ttime')
    print('=======================================================================')
    best_loss = np.inf
    for epoch in range(max_epochs):
        model.train()
        t0 = time()
        running_loss = 0.0
        for i, batch in enumerate(trainloader, 1):
            optimizer.zero_grad()
            X_i, y_i = batch
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
            model.eval()
            num_correct = 0
            valid_loss = 0
            for j, (X_j, y_j) in enumerate(validloader, 1):
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
    del model
    torch.cuda.empty_cache()
    model = _validation_load(pretrained_weights, train_embeddings, **model_params)
    return model


def predict(model, testloader):
    preds = []
    with torch.no_grad():
        model.eval()
        for X_i, y_i in testloader:
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
    scores['confusion_matrix'] = confusion_matrix(y_test, y_pred_test).tolist()
    return scores


def save_results(save_dir, scores, pretrained_weights, num_train, num_valid, num_test, batch_size, max_epochs, lr,
                 optimizer, criterion, train_embeddings, **model_params):
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
    results['train_embeddings'] = train_embeddings
    results['model_params'] = {**model_params}
    timestamp = '{:.0f}'.format(time())

    model_saves_dir = os.path.join(save_dir, 'model_saves')
    results_save_dir = os.path.join(save_dir, 'results')
    for directory in [save_dir, results_save_dir, model_saves_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    shutil.copyfile(VAL_SAVEPATH, os.path.join(model_saves_dir, '{}.pickle'.format(timestamp)))
    with open(os.path.join(results_save_dir, '{}.json'.format(timestamp)), 'w+') as outfile:
        json.dump(results, outfile)


def pretrain(pretrained_weights, save_dir='.', num_train=1024, valid_split=0.2, num_test=256, batch_size=16,
             max_epochs=100, lr=1e-3, optimizer=optim.Adam, criterion=nn.CrossEntropyLoss, train_embeddings=True,
             **model_params):
    trainset, validset, testset = _get_datasets(num_train, num_test, valid_split)
    output_dim = len(testset.LABELS)
    trainloader, validloader, testloader = [DataLoader(ds, batch_size=batch_size)
                                            for ds in (trainset, validset, testset)]
    model = model_cls(pretrained_weights, output_dim, train_embeddings=train_embeddings, **model_params).to(device)
    criterion = criterion(reduction='mean')
    optimizer = optimizer(model.parameters(), lr=lr)
    model = fit(model, optimizer, criterion, max_epochs, trainloader, validloader, pretrained_weights, train_embeddings,
                **model_params)
    y_pred_test = predict(model, testloader)
    scores = get_scores(testloader.dataset.y, y_pred_test)
    save_results(save_dir, scores, pretrained_weights, num_train, num_train, num_test, batch_size, max_epochs, lr,
                 optimizer, criterion, train_embeddings, **model_params)


if __name__ == '__main__':
    pretrain(os.path.join('pretrained_embeddings', 'word2vec', 'wikipedia-pubmed-and-PMC-w2v.bin'),
             train_embeddings=False, num_train=17,
             num_test=17, max_epochs=2)
