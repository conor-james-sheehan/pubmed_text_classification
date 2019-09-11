"""
'Smoke test' to make sure the training runs end to end. Uses a much smaller embedding matrix and only a fraction of the
training data.
"""
import os

import torch

from pubmed_text_classification.datasets import SupplementedAbstractSentencesDataset
from pubmed_text_classification.evaluate import evaluate
from pubmed_text_classification.model import TransitionModelConfig, WORD2VEC_EMBEDDING_DIM
from pubmed_text_classification.train import train

DUMMY_VOCAB_SIZE = 500
DUMMY_EMBEDDING = torch.rand(DUMMY_VOCAB_SIZE, WORD2VEC_EMBEDDING_DIM)


def main():
    dummy_word_to_ix = dict(unk=1)
    cfg = TransitionModelConfig(output_dim=SupplementedAbstractSentencesDataset.NUM_LABELS,
                                pretrained_embeddings=DUMMY_EMBEDDING,
                                word_to_ix=dummy_word_to_ix,
                                lstm_hidden_dim=128,
                                lstm_layers=1)
    batch_size = 64
    n_epochs = 2
    lr = .01
    savedir = os.path.join('../../results')
    model = train(config=cfg, num_train=1000, num_valid=100, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                  use_cuda=False)
    evaluate(model, savedir, batch_size, n_epochs, lr)


if __name__ == '__main__':
    main()
