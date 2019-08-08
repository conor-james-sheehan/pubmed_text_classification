import pandas as pd
from torch.utils.data import Dataset


class Pubmed20k(Dataset):

    LABELS = {'background': 0, 'objective': 1, 'methods': 2, 'results': 3, 'conclusions': 4}

    def __init__(self, path):
        self._load_txt(path)

    def _load_txt(self, path):
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
        self.dataframe = df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        return self.dataframe['sentence'].iloc[i], self.dataframe['label'].iloc[i]


if __name__ == '__main__':
    import os
    pubmed20k = \
        Pubmed20k('/home/conor/Downloads/HSLN-Joint-Sentence-Classification/data/PubMed_20k_RCT/test_clean.txt')
    pass