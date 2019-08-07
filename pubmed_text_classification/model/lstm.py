import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize


class TransitionMatrix(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.transition_probabilities = nn.Linear(1, output_dim, bias=False)

    def forward(self, previous_label):
        return self.transition_probabilities(previous_label.unsqueeze(1))


class NewModel(nn.Module):
    # TODO: rename

    def __init__(self, pretrained_weights, output_dim, hidden_dim=128, lstm_layers=1, dropout=0.5):
        super().__init__()
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_weights, binary=True)
        self.word_to_ix = {word: ix for ix, word in enumerate(word2vec.index2word)}
        # self.word_to_ix = {}
        weights = torch.FloatTensor(word2vec.vectors)
        # weights = torch.zeros(1, 100)
        self.embedding = nn.Embedding.from_pretrained(weights)

        self.lstm = nn.LSTM(input_size=weights.shape[1], hidden_size=hidden_dim//2, bidirectional=True,
                            num_layers=lstm_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_layer = nn.Linear(hidden_dim, output_dim)

        self.transition_matrix = TransitionMatrix(output_dim)

    def forward(self, X):
        sentence, last_label = X
        tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([self.word_to_ix.get(word, 0) for word in word_tokenize(s.lower())], dtype=torch.int64)
             for s in sentence])
        vec = self.embedding(tokens)
        drop = self.dropout(vec)

        lstm_out, (h, c) = self.lstm(drop)
        h = torch.cat([h[-2, :, :], h[-1, :, :]], dim=1)
        fc_out = F.relu(self.fc_layer(h))
        transition_out = self.transition_matrix(last_label)

        logits = fc_out + transition_out
        return logits


if __name__ == '__main__':
    import os
    from torch.utils.data import Dataset, DataLoader
    pretrained_weights = os.path.join('..', '..', 'pretrained_embeddings',
                                      'word2vec', 'wikipedia-pubmed-and-PMC-w2v.bin')
    model = NewModel(pretrained_weights, output_dim=5)
    last_label = torch.FloatTensor([0, 1, 2, 3])
    sentences = ['oxygen carbon silicon', 'dog cat mouse', 'flu cancer', 'biology chemistry']
    labels = [1, 2, 3, 4]

    class MyDataSet(Dataset):

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return [sentences[i], last_label[i]], labels[i]


    dataloader = DataLoader(MyDataSet(), batch_size=2)
    for X, y in dataloader:
        yhat = model(X)
