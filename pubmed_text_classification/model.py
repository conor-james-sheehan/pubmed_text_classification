import json
from zipfile import ZipFile

import gensim
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
from pkg_resources import resource_filename

WORD2VEC_VOCAB_SIZE = 5443656
WORD2VEC_EMBEDDING_DIM = 200

word_to_ix_path = resource_filename('pubmed_text_classification', 'word_to_ix.json.zip')
with ZipFile(word_to_ix_path, 'r') as zipfile:
    with zipfile.open('word_to_ix.json', 'r') as infile:
        WORD_TO_IX = json.load(infile)

nltk.download('punkt')
use_cuda = torch.cuda.is_available()
t = torch.cuda if use_cuda else torch
device = 'cuda:0' if use_cuda else 'cpu'


class TransitionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if isinstance(config.pretrained_embeddings, str):
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(config.pretrained_embeddings, binary=True)
            weights = torch.FloatTensor(word2vec.vectors)
        else:
            weights = config.pretrained_embeddings
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=not config.train_embeddings)
        self.lstm = nn.LSTM(input_size=WORD2VEC_EMBEDDING_DIM, hidden_size=config.lstm_hidden_dim // 2,
                            bidirectional=True, num_layers=config.lstm_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.output_dim = config.output_dim
        assert len(config.final_hidden_dims) == 2
        hdim1, hdim2 = config.final_hidden_dims
        self.final_mlp = nn.Sequential(nn.Linear(2 + config.lstm_hidden_dim, hdim1),
                                       nn.ReLU(),
                                       nn.Linear(hdim1, hdim2),
                                       nn.ReLU(),
                                       nn.Linear(hdim2, config.output_dim))

    def forward(self, X):
        sentence, sentence_num, total_sentences = X

        tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([self.config.word_to_ix.get(word, self.config.word_to_ix.get('unk'))
                           for word in word_tokenize(s.lower())],
                          dtype=torch.int64)
             for s in sentence]).to(device)
        vec = self.embedding(tokens)
        drop = self.dropout(vec)
        lstm_out, (h, c) = self.lstm(drop)
        elem_max, _ = lstm_out.max(dim=0)  # elementwise max over sequence length
        sentence_info = torch.cat([sentence_num.unsqueeze(dim=1), total_sentences.unsqueeze(dim=1)], dim=1) \
            .float().to(device)
        mlp_in = torch.cat([elem_max, sentence_info], dim=1)
        logits = self.final_mlp(mlp_in)
        return logits


class TransitionModelConfig:

    def __init__(self, output_dim, pretrained_embeddings='../pretrained_embeddings/wikipedia-pubmed-and-PMC-w2v.bin',
                 lstm_hidden_dim=512, lstm_layers=2, dropout=0.5, final_hidden_dims=(256, 128),
                 train_embeddings=False, word_to_ix=None):
        self.output_dim = output_dim
        self.pretrained_embeddings = pretrained_embeddings
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.final_hidden_dims = final_hidden_dims
        self.train_embeddings = train_embeddings
        self.word_to_ix = word_to_ix or WORD_TO_IX

    def to_json(self, path):
        with open(path, 'w+') as outfile:
            json.dump(self.__dict__, outfile)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as infile:
            _vars = json.load(infile)
        _vars['pretrained_embeddings'] = None
        return cls(**_vars)


def load_model(path, config):
    model = TransitionModel(config)
    model.load_state_dict(torch.load(path))
    return model.to(device)
