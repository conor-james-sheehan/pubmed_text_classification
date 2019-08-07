import os
from functools import reduce
import json
import xmltodict
import pandas as pd

ART_CORPUS_DIR = os.path.join('data', 'ART_Corpus')
DATASET_DIR = os.path.join('datasets', 'art')
TEST_SIZE = 0.2
os.chdir('..')

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)


def process_xml(xml_fpath):
    def extract_title(xml_dict):
        title = xml_dict['PAPER']['TITLE']['s']['annotationART']
        return {'sentence_type': 'title', 'sentence': title['#text'], 'label': title['@type']}

    def extract_sentences(xml_iter, sentence_type):
        sentences = []
        for xml_sentence in xml_iter:
            annotation = xml_sentence['annotationART']
            sentence = {'sentence_type': sentence_type, 'label': annotation['@type']}
            try:
                sentence['sentence'] = annotation['#text']
            except KeyError:
                try:
                    sentence['sentence'] = annotation['EQN']['#text']
                except KeyError:
                    continue

            sentences.append(sentence)
        return sentences

    def extract_section(section):
        # assert all([k in section for k in ['@DEPTH', 'HEADER', 'P']])

        if 'DIV' in section:
            # contains subsection
            subsection_sentences = [extract_section(subsection) for subsection in section['DIV']]
            subsection_sentences = reduce(lambda x, y: x + y, subsection_sentences)
        else:
            subsection_sentences = []

        def _extract_section(section):
            sentences = []
            for paragraph in section:
                sentences += extract_paragraph(paragraph)
            return sentences

        if 'P' in section:
            section = section['P']
            try:
                sentences = _extract_section(section)
            except TypeError:
                # section is not a list, i.e. just a single paragraph
                sentences = _extract_section([section])
        else:
            sentences = []
        return sentences + subsection_sentences

    def extract_paragraph(paragraph):
        try:
            if 's' not in paragraph:
                sentences = []
            else:
                sentences = extract_sentences(paragraph['s'], 'body')
        except TypeError:
            # paragraph is just a single sentence
            sentences = extract_sentences([paragraph['s']], 'body')
        return sentences

    def extract_abstract(xml_dict):
        abstract = xml_dict['PAPER']['ABSTRACT']['s']
        try:
            sentences = extract_sentences(abstract, 'abstract')
        except TypeError:
            sentences = extract_sentences([abstract], 'abstract')
        return sentences

    def extract_body(xml_dict):
        body = xml_dict['PAPER']['BODY']['DIV']
        sentences = []
        try:
            for section in body:
                sentences += extract_section(section)
        except TypeError:
            sentences = extract_section(body)
        return sentences

    with open(xml_fpath) as xml_file:
        xml_dict = xmltodict.parse(xml_file.read())

    title_sentence = extract_title(xml_dict)
    abstract_sentences = extract_abstract(xml_dict)
    body_sentences = extract_body(xml_dict)
    paper_sentences_df = pd.DataFrame([title_sentence] + abstract_sentences + body_sentences)
    return paper_sentences_df


def save_dataset(corpus):
    labels = corpus['label'].unique()
    label_mapping = dict(zip(labels, range(len(labels))))

    with open(os.path.join(DATASET_DIR, 'labels_mapping.json'), 'w+') as outfile:
        json.dump(label_mapping, outfile)

    def _get_numerical_label(label_str):
        return label_mapping[label_str]

    dataset = corpus['sentence'].to_frame()
    dataset['label'] = corpus['label'].apply(_get_numerical_label)
    num_test = int(len(dataset)*TEST_SIZE)
    train = dataset.iloc[:-num_test, :]
    test = dataset.iloc[-num_test:, :]
    train.to_csv(os.path.join(DATASET_DIR, 'train.csv'))
    test.to_csv(os.path.join(DATASET_DIR, 'test.csv'))


def main():
    dfs = []
    for folder in os.listdir(ART_CORPUS_DIR):
        folder_path = os.path.join(ART_CORPUS_DIR, folder)
        if os.path.isdir(folder_path):
            for xml_fname in os.listdir(folder_path):
                xml_fpath = os.path.join(folder_path, xml_fname)
                df = process_xml(xml_fpath)
                dfs.append(df)
    corpus = pd.concat(dfs, axis=0)
    corpus.to_csv(os.path.join(ART_CORPUS_DIR, 'corpus.csv'))
    save_dataset(corpus)


if __name__ == '__main__':
    main()
