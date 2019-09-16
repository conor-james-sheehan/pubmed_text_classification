from argparse import ArgumentParser
import sys
import os
sys.path += list(map(os.path.abspath, ['..', '../..', '../../..']))
from pubmed_text_classification.train import train
from pubmed_text_classification.datasets import SupplementedAbstractSentencesDataset
from pubmed_text_classification.evaluate import evaluate
from pubmed_text_classification.model import TransitionModelConfig

parser = ArgumentParser()
parser.add_argument('--pretrained_embeddings', type=str,
                    default='../../pretrained_embeddings/wikipedia-pubmed-and-PMC-w2v.bin',
                    help='Path to file containing pretrained word2vec weights in binary format.')
parser.add_argument('--pretrained_model', default=None, type=str,
                    help='Path to binary save file of a previously trained model. '
                         'If not specified, creates a new model to use.')
parser.add_argument('--savedir', default='../../results', type=str,
                    help='Path to directory wherein to save results and models.')
parser.add_argument('--n_epochs', default=100, type=int, help='Number of epochs to train for.')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')


def main():
    cmd_args = parser.parse_args()
    pretrained_embdeddings = cmd_args.pretrained_embeddings
    pretrained_model = cmd_args.pretrained_model
    savedir = cmd_args.savedir
    n_epochs = cmd_args.n_epochs
    batch_size = cmd_args.batch_size
    lr = cmd_args.lr
    config = TransitionModelConfig(SupplementedAbstractSentencesDataset.NUM_LABELS,
                                   pretrained_embeddings=pretrained_embdeddings)
    model = train(config,  model_path=pretrained_model, n_epochs=n_epochs, batch_size=batch_size, lr=lr)
    evaluate(model, savedir, batch_size, n_epochs, lr)


if __name__ == '__main__':
    main()
