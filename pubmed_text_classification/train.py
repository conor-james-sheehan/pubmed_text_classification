import os
from time import time
import pickle
from argparse import ArgumentParser
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from skorch import NeuralNet
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from pubmed_text_classification.model import BioBertMultiLabelClassifier, bert_tokenizer
from dataset_creation import TRAIN_FPATH, TEST_FPATH, LABELS

MODEL_SAVEDIR = 'model_saves'
SCORES_SAVEDIR = 'scores'

parser = ArgumentParser(description='train and test BioBert on the pubmed glycobiology abstract corpus')
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_epochs', default=2, type=int)


def tokenize_input(X, long_constructor):
    X = X.values.tolist()
    X = map(bert_tokenizer.encode, X)
    X = list(map(long_constructor, X))
    X = pad_sequence(X)
    return X.t()


def preprocess_dataset(ds, float_constructor, long_constructor):
    X = ds['sentence']
    X = tokenize_input(X, long_constructor)
    y = ds.iloc[:, 1:]
    y = float_constructor(y.values)
    assert len(X) == len(y)
    return X, y


def get_scores(y_test, y_pred_test):
    y_test = y_test.numpy()
    y_pred_test = y_pred_test.numpy()
    assert y_test.shape() == y_pred_test.shape()
    all_scores = []
    for i in range(y_test.shape[1]):
        scores = {}
        for score_name, score_func in [('accuracy', accuracy_score), (f1_score, 'f1_score'), ('auc', roc_auc_score)]:
            scores[score_name] = score_func(y_test[:, i], y_pred_test[:, i])
        all_scores.append(scores)
    return pd.DataFrame(all_scores, index=LABELS)


def save_model(model, timestamp):
    model_savepath = os.path.join(MODEL_SAVEDIR, '{}.pickle'.format(timestamp))
    with open(model_savepath, 'wb+') as outfile:
        pickle.dump(model, outfile)


def save_scores(scores, timestamp):
    scores_savepath = os.path.join(SCORES_SAVEDIR, '{}.csv'.format(timestamp))
    scores.to_csv(scores_savepath)


def train_model(device, batch_size, max_epochs, float_constructor, long_constructor):
    train = pd.read_csv(TRAIN_FPATH, index_col=0)
    test = pd.read_csv(TEST_FPATH, index_col=0)
    X_train, y_train = preprocess_dataset(train, float_constructor, long_constructor)
    X_test, y_test = preprocess_dataset(test, float_constructor, long_constructor)

    model = BioBertMultiLabelClassifier(output_dim=y_train.shape[1]).to(device)
    model = NeuralNet(model, optimizer=optim.Adam, criterion=nn.BCEWithLogitsLoss, max_epochs=max_epochs,
                      batch_size=batch_size, device=device)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    scores = get_scores(y_test, y_pred_test)

    timestamp = time()
    save_scores(scores, timestamp)
    save_model(model, timestamp)


def main():
    cmd_args = parser.parse_args()
    device = cmd_args.parse_args
    batch_size = cmd_args.batch_size
    max_epochs = cmd_args.max_epochs

    if device == 'cpu':
        float_constructor = torch.FloatTensor
        long_constructor = torch.LongTensor
    else:
        float_constructor = torch.cuda.FloatTensor
        long_constructor = torch.cuda.LongTensor

    train_model(device, batch_size, max_epochs, float_constructor, long_constructor)


if __name__ == '__main__':
    main()
