import json
from zipfile import ZipFile

from pubmed_text_classification.evaluate import get_pretrained_model, classify
from pubmed_text_classification.train import get_device

sentences_fname = 'GlyCosmos600/sentences.json'


def predict_json(zip_path, pretrained_path, output_path):
    device = get_device(use_cuda=True)
    model = get_pretrained_model(pretrained_path, device)

    with ZipFile(zip_path, 'r') as zipfile:
        with zipfile.open(sentences_fname, 'r') as sentences_file:
            sentences_json = json.load(sentences_file)

    sentences_dict = {x[0]: x[1] for x in sentences_json['data']}
    abstracts_dict = {}
    for key, sentence in sentences_dict.items():
        info = key.split('-')[1]
        abstract_id, _ = info.split('/')
        if abstract_id not in abstracts_dict:
            abstracts_dict[abstract_id] = []
        abstracts_dict[abstract_id].append(sentence)

    labels = []
    for abstract, sentences in abstracts_dict.items():
        labels += classify(abstract_sentences=sentences, model=model)

    for i in range(len(sentences_json['data'])):
        sentences_json['data'][i][1] = labels[i]

    with open(output_path, 'w') as outfile:
        json.dump(sentences_json, outfile)
