import os
from collections import Mapping
from tempfile import gettempdir
from time import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from pubmed_text_classification.model import TransitionModel, load_model, TransitionModelConfig
from pubmed_text_classification.datasets import SupplementedAbstractSentencesDataset

VAL_SAVEPATH = os.path.join(gettempdir(), 'model')  # temporary location to save best model during validation


def get_device(use_cuda):
    # TODO: move to a util file
    device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'
    return device


def _fit(model, optimizer, criterion, num_epochs, trainloader, validloader, device):
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
    config = model.config
    del model
    torch.cuda.empty_cache()
    model = load_model(VAL_SAVEPATH, config)
    return model


def _get_model(model_path, config):
    if config is None:
        # create config w/ default options
        config = TransitionModelConfig(SupplementedAbstractSentencesDataset.NUM_LABELS)
    else:
        if isinstance(config, str):
            # assume config is a path to a .json config
            config = TransitionModelConfig.from_json(config)
        if isinstance(config, Mapping):
            # conifg is kwargs to pass to config class init
            config = TransitionModelConfig(**config)

    if model_path is None:
        # make new model
        model_path = TransitionModel(config)
    else:
        model_path = load_model(model_path, config)
    return model_path
        

def train(config=None, model_path=None, num_train=None, num_valid=None, batch_size=256,
          n_epochs=100, lr=1e-3, optimizer=optim.Adam, criterion=nn.CrossEntropyLoss, use_cuda=True):
    device = get_device(use_cuda)
    print('Running on {}'.format(device))
    trainset = SupplementedAbstractSentencesDataset.from_txt('train', num_load=num_train)
    validset = SupplementedAbstractSentencesDataset.from_txt('dev', num_load=num_valid)
    trainloader, validloader = [DataLoader(ds, batch_size=batch_size) for ds in (trainset, validset)]
    model = _get_model(model_path, config).to(device)
    criterion = criterion(reduction='mean')
    optimizer = optimizer(model.parameters(), lr=lr)
    model = _fit(model, optimizer, criterion, n_epochs, trainloader, validloader, device)
    return model
