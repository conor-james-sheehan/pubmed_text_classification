from argparse import ArgumentParser
import torch
import sys
import os

sys.path += list(map(os.path.abspath, ['..', '../..', '../../..']))

from pubmed_text_classification.evaluate import rolling_classify_csv, Results
from pubmed_text_classification.model import TransitionModelConfig, load_model

parser = ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='Path to csv file for prediction.')
parser.add_argument('--savedir', default='../../predictions', type=str,
                    help='Where to save the new csv file containing the model\'s prediction for the dataset.')
parser.add_argument('--pretrained_path', default=None, type=str,
                    help='To a results directory containing a saved model & config. If unspecified, will use the '
                         'latest folder in /results')


def main():
    cmd_args = parser.parse_args()
    data_path = cmd_args.data_path
    savedir = cmd_args.savedir
    pretrained_path = cmd_args.pretrained_path

    if pretrained_path is None:
        results_dir = '../../results'
        results = os.listdir(results_dir)
        sorted_results = list(sorted(results, key=int))
        last_result = sorted_results[-1]
        pretrained_path = os.path.join(results_dir, last_result)

    assert all([fname in os.listdir(pretrained_path)
                for fname in [getattr(Results, attr+'_FNAME') for attr in ('MODEL', 'CONFIG')]]), \
        'directory {} does not contain the required files'.format(pretrained_path)
    config = TransitionModelConfig.from_json(os.path.join(pretrained_path, Results.CONFIG_FNAME))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model(os.path.join(pretrained_path, Results.MODEL_FNAME), config).to(device)
    new_ds = rolling_classify_csv(model, data_path)
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    csv_name = (data_path.split('/')[-1]).split('.csv')[0]
    csv_name += '_predicted.csv'
    save_path = os.path.join(savedir, csv_name)
    new_ds.to_csv(save_path)


if __name__ == '__main__':
    main()
