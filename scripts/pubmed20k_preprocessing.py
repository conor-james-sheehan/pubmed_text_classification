import os
import pandas as pd
import json


DATA_DIR = os.path.join('data', 'pubmed20k')


def load_file(fname):
    fpath = os.path.join(DATA_DIR, fname)

    with open(fpath, 'r') as infile:
        file_contents = infile.readlines()

    df = pd.DataFrame(map(json.loads, file_contents))
    df = df.drop('metadata', axis=1)
    return df


def save_labels(labels_mapping, folder):

    with open(os.path.join(folder, 'labels_mapping.json'), 'w+') as outfile:
        json.dump(labels_mapping, outfile)


def numericise_and_rename(df, labels_mapping):

    def _get_numerical_label(label):
        return labels_mapping[label]

    df['label'] = df['label'].apply(_get_numerical_label)
    df.rename({'text': 'sentence'}, axis=1)
    return df


def main():
    os.chdir('..')
    train = load_file('train.txt')
    labels = train['label'].unique()
    test = load_file('test.txt')

    labels_mapping = dict(zip(labels, range(len(labels))))
    train = numericise_and_rename(train, labels_mapping)
    test = numericise_and_rename(test, labels_mapping)

    dataset_dir = os.path.join('datasets', 'pubmed20k')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    save_labels(labels_mapping, dataset_dir)
    train.to_csv(os.path.join(dataset_dir, 'train.csv'))
    test.to_csv(os.path.join(dataset_dir, 'test.csv'))


if __name__ == '__main__':
    main()
