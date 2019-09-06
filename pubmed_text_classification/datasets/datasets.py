import os
import pandas as pd
from torch.utils.data import Dataset
from pkg_resources import resource_filename


class AbstractSentencesDataset(Dataset):
    LABELS = {'background': 0, 'objective': 1, 'methods': 2, 'results': 3, 'conclusions': 4}
    NUM_LABELS = len(LABELS)
    COLUMNS = {'abstract', 'sentence', 'label'}

    def __init__(self, dataframe):
        self.dataframe = dataframe

    @classmethod
    def from_txt(cls, set, num_load=None):
        assert set in ('train', 'test', 'dev')
        path = os.path.join(resource_filename('pubmed_text_classification', 'datasets'),\
                            'pubmed20k', '{}_clean.txt'.format(set))
        with open(path, 'r') as infile:
            txt = infile.read()
        # split per abstract
        txt = txt.split('###')
        # split lines
        txt = [abstract.split('\n') for abstract in txt]
        # collect into dict[abs_key] = list of abs sentences
        txt = {l[0]: l[1:] for l in txt}

        def _extract_df(abs_key, abstract):
            rows = []
            for sentence in abstract:
                try:
                    label, sentence = sentence.split('\t')
                except ValueError:
                    # line just contains ''
                    continue
                label = cls.LABELS[label.lower()]
                row = {'label': label, 'sentence': sentence, 'abstract': abs_key}
                rows.append(row)
            return pd.DataFrame(rows)

        dfs = [_extract_df(abs_key, abstract) for abs_key, abstract in txt.items()]
        df = pd.concat(dfs, axis=0)
        if num_load is not None:
            df = df.iloc[:num_load, :]
        return cls(df)

    @classmethod
    def from_csv(cls, path, **kwargs):
        df = pd.read_csv(path, **kwargs)
        assert set(df.columns) > cls.COLUMNS,\
            'csv file must have the following columns to be loaded as a dataset: {}'.format(list(cls.COLUMNS))
        return cls(df)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        return self.dataframe['sentence'].iloc[i], self.dataframe['label'].iloc[i]


class SupplementedAbstractSentencesDataset(AbstractSentencesDataset):

    def __init__(self, dataframe):
        super().__init__(dataframe)
        gb = self.dataframe.groupby('abstract')
        prev_labels = []
        for abstract in gb.groups:
            abs_df = gb.get_group(abstract)
            prev_labels.append(abs_df['label'].shift(1).fillna(-1))
        prev_labels = pd.concat(prev_labels, axis=0)
        self.dataframe['previous_label'] = prev_labels.values

    def __getitem__(self, i):
        sentence, label = super().__getitem__(i)
        prev_label = self.dataframe['previous_label'].iloc[i]
        X = sentence, prev_label
        return X, label
