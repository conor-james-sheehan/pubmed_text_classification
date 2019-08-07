import os
from pytorch_transformers.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--weights_dir')


def convert(weights_dir):
    model_fname = 'pytorch_model.bin'
    model_fpath = os.path.join(weights_dir, model_fname)
    config_fpath = os.path.join(weights_dir, 'bert_config.json')
    convert_tf_checkpoint_to_pytorch(weights_dir + '/biobert_model.ckpt', config_fpath, model_fpath)


if __name__ == '__main__':
    weights_dir = parser.parse_args().weights_dir
    convert(weights_dir)
