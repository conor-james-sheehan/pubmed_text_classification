import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionMatrix(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.transition_probabilities = nn.Linear(1, output_dim, bias=False)

    def forward(self, previous_label):
        return self.transition_probabilities(previous_label)


class NewModel(nn.Module):
    # TODO: rename

    def __init__(self, pretrained_weights, output_dim, hidden_dim=128, lstm_layers=1, dropout=0.5):
        super().__init__()
        # TODO: get word_to_ix
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_weights)
        weights = torch.FloatTensor(self.word2vec.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(input_size=weights.shape[1], hidden_size=hidden_dim, bidirectional=True,
                            num_layers=lstm_layers)
        self.drop = nn.Dropout(dropout)
        self.fc_layer = nn.Linear(hidden_dim, output_dim)

        self.transition_matrix = TransitionMatrix(output_dim)

    def forward(self, X):
        sentence, last_label = X
        tokens = self.word2vec.tokenize(sentence)
        vec = self.embedding(tokens)
        lstm_out = self.lstm(vec)
        dropout = self.dropout(lstm_out)
        fc_out = F.relu(self.fc_layer(dropout))
        transition_out = self.transition_matrix(last_label)

        logits = fc_out + transition_out
        return logits
