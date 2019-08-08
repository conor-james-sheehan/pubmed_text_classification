import os
import pandas as pd
from torch.utils.data import Dataset


class Pubmed20k(Dataset):

    LABELS = {'background': 0, 'objective': 1, 'methods': 2, 'results': 3, 'conclusions': 4}

    def __init__(self, set, num_load=None):
        self._load_txt(set, num_load)

    def _load_txt(self, set, num_load):
        assert set in ('train', 'test')
        path = os.path.join('pubmed_text_classification', 'datasets', 'pubmed20k', '{}_clean.txt'.format(set))
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
                label = self.LABELS[label.lower()]
                row = {'label': label, 'sentence': sentence, 'abstract': abs_key}
                rows.append(row)
            return pd.DataFrame(rows)

        dfs = [_extract_df(abs_key, abstract) for abs_key, abstract in txt.items()]
        df = pd.concat(dfs, axis=0)
        if num_load is not None:
            df = df.iloc[:num_load, :]
        self.dataframe = df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        return self.dataframe['sentence'].iloc[i], self.dataframe['label'].iloc[i]


class PubMed20kPrevious(Pubmed20k):

    def __init__(self, set, num_load=None):
        super().__init__(set, num_load=num_load)
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
