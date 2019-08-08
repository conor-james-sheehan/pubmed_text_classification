import os
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertConfig, BertModel
from sklearn.base import TransformerMixin, BaseEstimator
from torch.nn.utils.rnn import pad_sequence

from pubmed_text_classification.scripts.convert_bert_pytorch import convert

BERT_DIM = 768
MAX_BERT_SEQ_LEN = 512

use_cuda = torch.cuda.is_available()
t = torch.cuda if use_cuda else torch
device = 'cuda:0' if use_cuda else 'cpu'


class TokenizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pretrained_weights):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X_t = X.tolist()
        X_t = map(self.tokenizer.encode, X)
        X_t = [X_i[:MAX_BERT_SEQ_LEN] for X_i in X_t]
        X_t = list(map(torch.LongTensor, X_t))
        X_t = pad_sequence(X_t)
        return X_t.t()


def _get_custom_bert(pretrained_weights):
    model_fname = 'pytorch_model.bin'
    if model_fname not in os.listdir(pretrained_weights):
        convert(pretrained_weights)
    model_fpath = os.path.join(pretrained_weights, model_fname)
    config_fpath = os.path.join(pretrained_weights, 'bert_config.json')
    config = BertConfig.from_json_file(config_fpath)
    custom_bert = BertModel(config)
    state_dict = torch.load(model_fpath)

    def _remove_prefix(string):
        prefix = 'bert.'
        if string.startswith(prefix):
            string = string[len(prefix):]
        return string

    state_dict = {_remove_prefix(k): v for k, v in state_dict.items() if not k.startswith('cls')}
    custom_bert.load_state_dict(state_dict)
    return custom_bert


class BaseBertExtensionModel(nn.Module):

    def __init__(self, pretrained_weights, train_bert):
        super().__init__()
        try:
            self.bert = _get_custom_bert(pretrained_weights)
        except FileNotFoundError:
            self.bert = BertModel.from_pretrained(pretrained_weights)

        if train_bert:
            assert all([p.requires_grad for p in self.bert.parameters()])
        else:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, *input):
        raise NotImplementedError


class BertClassifier(BaseBertExtensionModel):

    def __init__(self, pretrained_weights, output_dim, dropout=0.2, train_embeddings=False):
        super().__init__(pretrained_weights, train_embeddings)
        self.dropout = nn.Dropout(p=dropout)
        self.out_layer = nn.Linear(BERT_DIM, output_dim)
        self.tokenizer = TokenizerTransformer(pretrained_weights)

    def forward(self, X):
        torch.cuda.empty_cache()
        X_t = self.tokenizer.fit_transform(X).to(device)
        _, bert_out = self.bert(X_t)
        dropped = self.dropout(bert_out)
        logits = self.out_layer(dropped)
        return logits
