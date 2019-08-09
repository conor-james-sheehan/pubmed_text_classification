import json
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
use_cuda = torch.cuda.is_available()
t = torch.cuda if use_cuda else torch
device = 'cuda:0' if use_cuda else 'cpu'


def _to_one_hot(labels, output_dim):
    one_hot = torch.zeros(len(labels), output_dim + 1)
    for i, j in enumerate(map(int, labels.numpy().tolist())):
        one_hot[i, j] = 1.0
    return one_hot.to(device)


class TransitionMatrix(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.transition_probabilities = nn.Linear(output_dim+1, output_dim, bias=False)

    def forward(self, previous_label):
        """

        :param previous_label: one-hot vector of previous label
        :return:
        """
        return self.transition_probabilities(previous_label)


class TransitionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # todo: implement train_embeddings behaviour
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(config.pretrained_embeddings, binary=True)
        self.word_to_ix = {word: ix for ix, word in enumerate(word2vec.index2word)}
        weights = torch.FloatTensor(word2vec.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights)

        self.lstm = nn.LSTM(input_size=weights.shape[1], hidden_size=config.hidden_dim//2, bidirectional=True,
                            num_layers=config.lstm_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_layer = nn.Linear(config.hidden_dim, config.output_dim)
        self.output_dim = config.output_dim

        self.transition_matrix = TransitionMatrix(self.output_dim)

    def forward(self, X):
        sentence, last_label = X
        # TODO: sort out what to do w/ unknown words
        tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([self.word_to_ix.get(word, 0) for word in word_tokenize(s.lower())], dtype=torch.int64)
             for s in sentence]).to(device)
        vec = self.embedding(tokens)
        drop = self.dropout(vec)

        lstm_out, (h, c) = self.lstm(drop)
        h = torch.cat([h[-2, :, :], h[-1, :, :]], dim=1)
        fc_out = F.relu(self.fc_layer(h))

        last_label = _to_one_hot(last_label, output_dim=self.output_dim)
        transition_out = self.transition_matrix(last_label)

        logits = fc_out + transition_out

        return logits


class TransitionModelConfig:

    def __init__(self, output_dim, pretrained_embeddings='../pretrained_embeddings/wikipedia-pubmed-and-PMC-w2v.bin',
                 hidden_dim=512, lstm_layers=2, dropout=0.5, train_embeddings=False):
        self.output_dim = output_dim
        self.pretrained_embeddings = pretrained_embeddings
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.train_embeddings = train_embeddings

    def to_json(self, path):
        with open(path, 'w+') as outfile:
            json.dump(self.__dict__, outfile)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as infile:
            _vars = json.load(infile)
        return cls(**_vars)


def load_model(path, config):
    # TODO: perhaps an api function to load model w/o/ knowing what its params were; perhaps a config for the model
    model = TransitionModel(config)
    model.load_state_dict(torch.load(path))
    return model.to(device)
