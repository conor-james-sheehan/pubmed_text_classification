import os
import numpy as np
import pandas as pd

LABELS = ['Introduction', 'Hypothesis', 'Method', 'Results', 'Conclusion', 'Other']
DATASET_DIR = os.path.join('datasets', 'pubmed')
TRAIN_FPATH = os.path.join(DATASET_DIR, 'train.csv')
TEST_FPATH = os.path.join(DATASET_DIR, 'test.csv')
TEST_SIZE = 0.2

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)


def label_to_encoded(label):
    encoded_label = np.zeros((len(LABELS),))
    labels = label.split(',')
    for i in map(int, labels):
        encoded_label[i] = 1
    return encoded_label


def main():
    corpus_fname = 'pubmed_glycobiology_corpus.csv'
    corpus_fpath = os.path.join('data', corpus_fname)
    corpus = pd.read_csv(corpus_fpath, index_col='id').dropna()
    y = list(map(label_to_encoded, corpus['label']))
    y = pd.DataFrame(y, columns=LABELS)
    X = corpus.reset_index()['sentence']
    dataset = pd.concat([X, y], axis=1)
    num_test = int(len(dataset)*TEST_SIZE)
    train = dataset.iloc[:-num_test, :]
    test = dataset.iloc[-num_test:, :]
    train.to_csv(TRAIN_FPATH)
    test.to_csv(TEST_FPATH)


if __name__ == '__main__':
    main()
