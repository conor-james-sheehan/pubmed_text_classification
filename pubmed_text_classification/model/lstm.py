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
    for i, j in enumerate(labels):
        one_hot[i, j] = 1
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


class TransitonModel(nn.Module):

    def __init__(self, pretrained_weights, output_dim, hidden_dim=128, lstm_layers=1, dropout=0.5,
                 train_embeddings=True):
        super().__init__()
        # todo: implement train_embeddings behaviour
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_weights, binary=True)
        self.word_to_ix = {word: ix for ix, word in enumerate(word2vec.index2word)}
        weights = torch.FloatTensor(word2vec.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights)

        self.lstm = nn.LSTM(input_size=weights.shape[1], hidden_size=hidden_dim//2, bidirectional=True,
                            num_layers=lstm_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_layer = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

        self.transition_matrix = TransitionMatrix(output_dim)

    def forward(self, X):
        sentence, last_label = X
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
