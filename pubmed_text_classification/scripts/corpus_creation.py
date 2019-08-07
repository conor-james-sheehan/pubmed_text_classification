import os
import pandas as pd
from abstract_extraction import PROJECT_DIR
from nltk import tokenize


def extract_sentence_numbers(index, row):
    numbered_sentences = []
    sentences = []
    try:
        sentences += tokenize.sent_tokenize(row['abstract'])
    except TypeError:
        # abstract is NaN
        return
    for i, sentence in enumerate(sentences):
        numbered_sentences.append({
            'id': index,
            'sentence_num': i,
            'sentence': sentence
        })
    return pd.DataFrame(numbered_sentences).set_index(['id', 'sentence_num'])


def main():
    abstracts = pd.read_csv(PROJECT_DIR + '/data/suitable_abstracts.csv', index_col='id')
    dfs = []
    for index, row in abstracts.iterrows():
        numbered_sentences = extract_sentence_numbers(index, row)
        if numbered_sentences is not None:
            dfs.append(numbered_sentences)

    corpus = pd.concat(dfs, axis=0)
    corpus.to_csv(os.path.join(PROJECT_DIR, 'data', 'corpus.csv'))


if __name__ == '__main__':
    main()
