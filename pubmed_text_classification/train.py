import os
from tempfile import gettempdir
from time import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from pubmed_text_classification.model import TransitonModel
from pubmed_text_classification.datasets import SupplementedAbstractSentencesDataset

VAL_SAVEPATH = os.path.join(gettempdir(), 'model')  # temporary location to save best model during validation
OUTPUT_DIM = len(SupplementedAbstractSentencesDataset.LABELS)  # number of classes (= 5)

use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
t = torch.cuda if use_cuda else torch
print('Running on {}'.format(device))


def _split_data(num_train, valid_split):
    # split the training set into training and validation
    trainset = SupplementedAbstractSentencesDataset.from_txt('train', num_load=num_train)
    num_valid = int(valid_split*len(trainset))
    num_train = len(trainset) - num_valid
    trainset, validset = random_split(trainset, [num_train, num_valid])
    return trainset, validset


def _fit(model, optimizer, criterion, num_epochs, trainloader, validloader, pretrained_weights, **model_params):
    print('epoch\t\ttrain_loss\tvalid_loss\tvalid_acc\ttime')
    print('=======================================================================')
    best_loss = np.inf
    for epoch in range(num_epochs):
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
            del X_i, y_i, y_hat_i, loss
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
                del X_j, y_j, y_hat_j, loss, predicted
                torch.cuda.empty_cache()
            valid_loss /= j
            running_loss /= i
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), VAL_SAVEPATH)
        accuracy = num_correct / len(validloader.dataset)
        dt = time() - t0
        print('{}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}'
              .format(epoch, running_loss, valid_loss, accuracy, dt))
    model = model.cpu()
    del model
    torch.cuda.empty_cache()
    model = _load_model(VAL_SAVEPATH, pretrained_weights, **model_params)
    return model


def _load_model(path, pretrained_weights, **model_params):
    # TODO: perhaps an api function to load model w/o/ knowing what its params were; perhaps a config for the model
    model = TransitonModel(pretrained_weights, OUTPUT_DIM, **model_params)
    model.load_state_dict(torch.load(path))
    return model.to(device)


def _get_model(model_path, pretrained_weights, **model_params):
    if model_path is None:
        model_path = TransitonModel(pretrained_weights, OUTPUT_DIM, **model_params)
    else:
        model_path = _load_model(model_path, pretrained_weights, **model_params)
    return model_path
        

def train(pretrained_weights, model_path=None, num_train=None, valid_split=0.2, batch_size=256,
          n_epochs=100, lr=1e-3, optimizer=optim.Adam, criterion=nn.CrossEntropyLoss,
          **model_params):
    trainset, validset = _split_data(num_train, valid_split)
    trainloader, validloader = [DataLoader(ds, batch_size=batch_size) for ds in (trainset, validset)]
    model = _get_model(model_path, pretrained_weights, **model_params)

    criterion = criterion(reduction='mean')
    optimizer = optimizer(model_path.parameters(), lr=lr)
    model = _fit(model, optimizer, criterion, n_epochs, trainloader, validloader, pretrained_weights,
                 **model_params)
    return model

    
if __name__ == '__main__':
    train(os.path.join('pretrained_embeddings', 'word2vec', 'wikipedia-pubmed-and-PMC-w2v.bin'),
          train_embeddings=False, num_train=17,
          num_test=17, n_epochs=2)
