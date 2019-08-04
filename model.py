import os
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertConfig, BertModel
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from torch.nn.utils.rnn import pad_sequence

WEIGHTS_DIR = 'biobert_v1.0_pubmed'
BERT_DIM = 768
MAX_BERT_SEQ_LEN = 512

model_fname = 'pytorch_model.bin'
model_fpath = os.path.join(WEIGHTS_DIR, model_fname)
config_fpath = os.path.join(WEIGHTS_DIR, 'bert_config.json')
use_cuda = torch.cuda.is_available()
t = torch.cuda if use_cuda else torch
print('Running on {}'.format('gpu' if use_cuda else 'cpu'))


class TokenizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pretrained_weights):
        if pretrained_weights.lower() == 'biobert':
            pretrained_weights = WEIGHTS_DIR
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    def fit(self, X, y=None):
        # X_list = X.tolist()
        # X_token = map(self.tokenizer.encode, X_list)
        # self.max_len_ = max(map(len, X_token))
        return self

    def transform(self, X):
        X_t = X.tolist()
        X_t = map(self.tokenizer.encode, X_t)
        X_t = [X_i[:MAX_BERT_SEQ_LEN] for X_i in X_t]
        X_t = list(map(torch.LongTensor, X_t))
        X_t = pad_sequence(X_t)
        return X_t.t()


def _get_biobert():
    config = BertConfig.from_json_file(config_fpath)
    biobert = BertModel(config)
    state_dict = torch.load(model_fpath)

    def _remove_prefix(string):
        prefix = 'bert.'
        if string.startswith(prefix):
            string = string[len(prefix):]
        return string

    state_dict = {_remove_prefix(k): v for k, v in state_dict.items() if not k.startswith('cls')}
    biobert.load_state_dict(state_dict)
    return biobert


class BertClassifier(nn.Module):

    def __init__(self, pretrained_weights, output_dim, dropout=0.2, train_bert=True):
        super().__init__()
        if pretrained_weights.lower() == 'biobert':
            self.bert = _get_biobert()
        else:
            self.bert = BertModel.from_pretrained(pretrained_weights)

        if train_bert:
            self.bert.eval()
            assert all([not p.requires_grad for p in self.bert.parameters()])
        else:
            assert all([p.requires_grad for p in self.bert.parameters()])
        self.dropout = nn.Dropout(p=dropout)
        self.out_layer = nn.Linear(BERT_DIM, output_dim)

    def forward(self, X):
        torch.cuda.empty_cache()
        X = t.LongTensor(X)
        _, h = self.bert(X)
        h_drop = self.dropout(h)
        logits = self.out_layer(h_drop)
        return logits


def get_bert_model_pipeline(pretrained_weights, output_dim, dropout=0.5, device='cpu', *args, **kwargs):
    clf = BertClassifier(pretrained_weights, output_dim, dropout=dropout).to(device)
    clf = NeuralNetClassifier(clf, device=device, *args, **kwargs)
    tokenizer = TokenizerTransformer(pretrained_weights)
    model = Pipeline([
        ('tokenizer', tokenizer),
        ('classifier', clf)
    ])
    return model


if __name__ == '__main__':
    import pandas as pd

    MAX_EPOCHS = 2
    BATCH_SIZE = 10
    NUM_TRAIN = 25000

    DATASET_DIR = os.path.join('datasets', 'art')
    train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'), index_col=0)
    test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'), index_col=0)
    train = train.iloc[:NUM_TRAIN, :]

    def preprocess_data(ds):
        X = ds['sentence'].values
        y = ds['label'].values
        return X, y


    X_train, y_train = preprocess_data(train)
    X_test, y_test = preprocess_data(test)
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    n_batches = NUM_TRAIN/BATCH_SIZE
    X_train = np.split(X_train, n_batches)
    y_train = np.split(y_train, n_batches)

    tokenizer = TokenizerTransformer('biobert')
    model = BertClassifier('biobert', 11)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(size_average=True)

    for epoch in range(MAX_EPOCHS):
        running_loss = 0.0
        for i, batch in enumerate(zip(X_train, y_train), 1):
            optimizer.zero_grad()
            X_i, y_i = batch
            X_i = tokenizer.fit_transform(X_i)
            X_i, y_i = map(t.LongTensor, [X_i, y_i])
            y_hat_i = model(X_i)
            loss = criterion(y_hat_i, y_i)
            loss.backward()
            optimizer.step()
            del X_i
            del y_i
            del y_hat_i
            torch.cuda.empty_cache()
            running_loss += loss.item()
            if i % 100 == 0:
                print('Finished batch {}, loss {}'.format(i, running_loss/n_batches))
