import json
import os
from time import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from pubmed_text_classification.datasets import SupplementedAbstractSentencesDataset


def _get_scores(y_test, y_pred_test):
    scores = {}
    scores['accuracy'] = accuracy_score(y_test, y_pred_test)
    scores['confusion_matrix'] = confusion_matrix(y_test, y_pred_test).tolist()
    return scores


class Results:

    def __init__(self, model, scores, valid_split, batch_size, n_epochs, lr):
        self.model = model
        self.scores = scores
        self.meta = dict(valid_split=valid_split, batch_size=batch_size, n_epochs=n_epochs, lr=lr)
        self.results_dir = None

    def save(self, savedir):
        timestamp = '{:.0f}'.format(time())
        self.results_dir = os.path.join(savedir, timestamp)
        os.makedirs(self.results_dir)
        self._save_model()
        self._save_config()
        self._save_meta()
        self._save_scores()

    def _save_model(self):
        model_savepath = os.path.join('model.pickle')
        torch.save(self.model.state_dict(), model_savepath)

    def _save_json(self, obj, fname):
        fpath = os.path.join(self.results_dir, fname)
        with open(fpath, 'w+') as outfile:
            json.dump(obj, outfile)

    def _save_config(self):
        config_path = os.path.join(self.results_dir, 'config.json')
        self.model.config.to_json(config_path)

    def _save_meta(self):
        self._save_json(self.meta, 'meta.json')

    def _save_scores(self):
        self._save_json(self.scores, 'score.json')


def _predict(model, testloader):
    preds = []
    with torch.no_grad():
        model.eval()
        for X_i, y_i in testloader:
            y_hat_i = model(X_i)
            _, predicted = torch.max(y_hat_i.data, 1)
            predicted = predicted.cpu()
            preds.append(predicted.numpy())
            del X_i, y_hat_i, predicted
            torch.cuda.empty_cache()
    return np.concatenate(preds)


def _load_test(test_path):
    if test_path is None:
        testset = SupplementedAbstractSentencesDataset.from_txt('test')
    else:
        testset = SupplementedAbstractSentencesDataset.from_csv(test_path)

    return testset


def evaluate(model, test_path, savedir, valid_split, batch_size, n_epochs, lr):
    testset = _load_test(test_path)
    testloader = DataLoader(testset, batch_size=batch_size)
    y_test = testset.dataframe['label'].values
    y_pred_test = _predict(model, testloader)
    scores = _get_scores(y_test, y_pred_test)
    results = Results(model, scores, valid_split, batch_size, n_epochs, lr)
    results.save(savedir)
    return y_pred_test, testset
