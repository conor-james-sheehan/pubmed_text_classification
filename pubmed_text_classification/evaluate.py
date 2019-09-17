import json
import os
import re
from time import time
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from pubmed_text_classification.datasets import SupplementedAbstractSentencesDataset
from pubmed_text_classification.model import SentenceModelConfig, load_model
from pubmed_text_classification.train import get_device

# TODO: docstrings on API functions


def _get_scores(y_test, y_pred_test):
    scores = {}
    scores['accuracy'] = accuracy_score(y_test, y_pred_test)
    scores['confusion_matrix'] = confusion_matrix(y_test, y_pred_test).tolist()
    return scores


class Results:
    MODEL_FNAME = 'model.pickle'
    CONFIG_FNAME = 'config.json'
    META_FNAME = 'meta.json'
    SCORES_FNAME = 'scores.json'

    def __init__(self, model, scores, batch_size, n_epochs, lr):
        self.model = model
        self.scores = scores
        self.meta = dict(batch_size=batch_size, n_epochs=n_epochs, lr=lr)
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
        model_savepath = os.path.join(self.results_dir, self.MODEL_FNAME)
        torch.save(self.model.state_dict(), model_savepath)

    def _save_json(self, obj, fname):
        fpath = os.path.join(self.results_dir, fname)
        with open(fpath, 'w+') as outfile:
            json.dump(obj, outfile)

    def _save_config(self):
        config_path = os.path.join(self.results_dir, self.CONFIG_FNAME)
        self.model.config.to_json(config_path)

    def _save_meta(self):
        self._save_json(self.meta, self.META_FNAME)

    def _save_scores(self):
        self._save_json(self.scores, self.SCORES_FNAME)


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


def evaluate(model, savedir, batch_size, n_epochs, lr):
    testset = SupplementedAbstractSentencesDataset.from_txt('test')
    testloader = DataLoader(testset, batch_size=batch_size)
    y_test = testset.dataframe['label'].values
    y_pred_test = _predict(model, testloader)
    scores = _get_scores(y_test, y_pred_test)
    results = Results(model, scores, batch_size, n_epochs, lr)
    results.save(savedir)


def classify(model, abstract_sentences):
    """

    :param model:
    :param abstract_sentences:
    :return:
    """
    abstract_sentences = list(map(_replace_digits, abstract_sentences))
    df = pd.DataFrame(columns=SupplementedAbstractSentencesDataset.COLUMNS)
    df['sentence'] = abstract_sentences
    df = df.fillna(-1)
    ds = SupplementedAbstractSentencesDataset(df)
    testloader = DataLoader(ds, batch_size=len(abstract_sentences))
    predicted = _predict(model, testloader).tolist()
    return predicted


def classify_from_pretrained(abstract_sentences, pretrained_path, use_cuda=True):
    device = get_device(use_cuda)
    model = get_pretrained_model(pretrained_path, device)
    return classify(model, abstract_sentences)


def _replace_digits(sentence):
    digit_regex = r'(\d+\.\d+|\d+)'
    return re.sub(digit_regex, '@', sentence)


def get_pretrained_model(path, device):
    config = SentenceModelConfig.from_json(os.path.join(path, 'config.json'))
    model = load_model(os.path.join(path, 'model.pickle'), config, device)
    return model
