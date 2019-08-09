from argparse import ArgumentParser
import sys
import os

sys.path += list(map(os.path.abspath, ['..', '../..', '../../..']))

from pubmed_text_classification.evaluate import evaluate, Results
from pubmed_text_classification.model import TransitionModel, TransitionModelConfig, load_model

parser = ArgumentParser()
parser.add_argument('--test_path', default=None, type=str,
                    help='Path to csv file to use for test dataset. '
                         'If unspecified, will use the pubmed20k dataset.')
parser.add_argument('--model_dir', default=None, type=str,
                    help='To a results directory containing a saved model & config. If unspecified, will use the '
                         'latest folder in /results')
parser.add_argument('--batch_size', default=256, type=int)


def main():
    cmd_args = parser.parse_args()
    pretrained_path = cmd_args.test_path
    test_path = cmd_args.test_path

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
    model = load_model(os.path.join(pretrained_path, Results.MODEL_FNAME), config)
    testset =

if __name__ == '__main__':
    main()
