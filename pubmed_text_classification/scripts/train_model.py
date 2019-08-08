from argparse import ArgumentParser

from pubmed_text_classification.train import train

parser = ArgumentParser()
parser.add_argument('--train_path', default=None,
                    help='Path to csv file to use for training dataset. '
                         'If unspecified, will use the pubmed20k dataset.')
parser.add_argument('--test_path', default=None,
                    help='Path to csv file to use for test dataset. '
                         'If unspecified, will use the pubmed20k dataset.')
parser.add_argument('--pretrained_weights', default='../pretrained_weights/wikipedia-pubmed-and-PMC-w2v.bin',
                    help='Path to file containing pretrained word2vec weights in binary format.')
parser.add_argument('--pretrained_model', default=None, help='Path to binary save file of a previously trained model. '
                                                             'If not specified, creates a new model to use.')
parser.add_argument('--savedir', default=None, help='Path to directory wherein to save results and models.')
parser.add_argument('--n_epochs', default=100, help='Number of epochs to train for.')
parser.add_argument('--batch_size', default=256)
parser.add_argument('--valid_split', default=0.2, help='Fraction of data to use for validation.')
parser.add_argument('--lr', default=0.01, help='Learning rate')


def main():
    cmd_args = parser.parse_args()
    train_path = cmd_args.train_path
    test_path = cmd_args.test_path
    pretrained_weights = cmd_args.pretrained_weights
    pretrained_model = cmd_args.pretrained_model
    savedir = cmd_args.save_dir
    n_epochs = cmd_args.n_epochs
    batch_size = cmd_args.batch_size
    valid_split = cmd_args.valid_split
    lr = cmd_args.lr
    model = train(pretrained_weights, model_path=pretrained_model, valid_split=valid_split, num_epochs=n_epochs,
                  batch_size=batch_size, lr=lr)


if __name__ == '__main__':
    main()
