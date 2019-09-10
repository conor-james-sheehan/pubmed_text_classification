"""
'Smoke test' to make sure the training runs end to end. Uses a much smaller embedding matrix and only a fraction of the
training data.
"""

import numpy as np
import torch

from pubmed_text_classification.datasets import SupplementedAbstractSentencesDataset
from pubmed_text_classification.model import TransitionModelConfig, WORD2VEC_EMBEDDING_DIM
from pubmed_text_classification.train import train

DUMMY_VOCAB_SIZE = 500
DUMMY_EMBEDDING = torch.rand(DUMMY_VOCAB_SIZE, WORD2VEC_EMBEDDING_DIM)


class DummyWord2IX:

    def __init__(self):
        self.choices = list(range(DUMMY_VOCAB_SIZE))

    def get(self, *args):
        return np.random.choice(self.choices)


def main():
    cfg = TransitionModelConfig(output_dim=SupplementedAbstractSentencesDataset.NUM_LABELS,
                                pretrained_embeddings=DUMMY_EMBEDDING,
                                word_to_ix=DummyWord2IX(),
                                lstm_hidden_dim=128,
                                lstm_layers=1)
    train(config=cfg, num_train=1000, num_valid=100, batch_size=64, n_epochs=10)


if __name__ == '__main__':
    main()
