import json
import os
import re
from time import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from pubmed_text_classification.datasets import SupplementedAbstractSentencesDataset, AbstractSentencesDataset


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


def rolling_classify_csv(model, fpath):
    """

    :param model:
    :type model: TransitionModel
    :param fpath: path to a .csv file containing an unlabelled dataset.
    :type fpath: str
    :return:
    """
    df = AbstractSentencesDataset.from_csv(fpath).dataframe
    df['predicted_label'] = np.nan
    gb = df.groupby('abstract')

    for abstract in gb.groups:
        abstract_df = gb.get_group(abstract)
        predicted_classes = rolling_classify(model, abstract_df['sentence'])
        for i, pred in enumerate(predicted_classes):
            df.loc[abstract_df.index[i], 'predicted_label'] = pred
    return df


def rolling_classify(model, sentences):
    """
    Classify sentences from a given abstract.

    :param model:
    :type model: TransitionModel
    :param sentences:
    type sentences: list[str]
    :return: predicted_labels
    :rtype: list[int]
    """
    sentences = map(_replace_digits, sentences)
    y = torch.FloatTensor([-1.0])
    predictions = []
    for sentence in sentences:
        X = [sentence], y
        probs = model(X)
        _, y = torch.max(probs.data, 1)
        y = y.cpu()
        predictions.append(y.item())
    return predictions


def _replace_digits(sentence):
    digit_regex = r'(\d+\.\d+|\d+)'
    return re.sub(digit_regex, '@', sentence)


if __name__ == '__main__':
    import pkg_resources
    import os
    fpath = os.path.join(pkg_resources.resource_filename('pubmed_text_classification', 'datasets'),
                         'pubmed-glyco', 'corpus.csv')

    class DummyModel:
        def __call__(self, X):
            return torch.rand((1, 5))

    rolling_classify_csv(DummyModel(), fpath)
    # rolling_predict()
    pass
