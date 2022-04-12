import os
import argparse
import json
import utilities


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Baseline script for transfer learning')
    parser.add_argument('--d', type=utilities.dir_path,
                        help='Directory where files are located (absolute dir)')
    parser.add_argument('--d_test', type=utilities.dir_path,
                        help='Directory where test files are located (absolute dir)')
    parser.add_argument('--d_valid', type=utilities.dir_path,
                        help='Directory where validation files are located (absolute dir)')
    parser.add_argument('--model', type=utilities.dir_path,
                        help='Directory where model is located (absolute dir)')
    parser.add_argument('--features', type=utilities.dir_path,
                        help='Directory where features are located (absolute dir)')
    parser.add_argument('--results', type=utilities.dir_path,
                        help='Directory where results should be stored (absolute dir)')
    parser.add_argument('--extract', dest='extract',
                        action='store_true', help='Extract features and store it')
    parser.add_argument('--cosine', dest='cosine',
                        action='store_true', help='Apply cosine distance metric')
    parser.add_argument('--mean', dest='mean', action='store_true',
                        help='Apply cosine distance on mean feature')
    parser.add_argument('--knn', dest='knn',
                        action='store_true', help='Apply kNN metric')
    parser.add_argument('--svm', dest='svm', action='store_true',
                        help='Apply Support Vector Machine')
    parser.add_argument('--k', type=int, dest='k', default=5,
                        help='Define k for kNN algorithm (Default: 5)')
    parser.add_argument('--step', type=int, dest='step', default=5,
                        help='Define step with which training set should be decreased (Default: k=5)')
    parser.add_argument('--max-size', type=int, dest='max_size', default=5,
                        help='Define maximum samples per class (Default: k=5)')
    parser.add_argument('--unbalanced', dest='unbalanced', action='store_true',
                        help='Define if dataset is unbalanced (Default: false)')
    parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                        help='Define if training should be stopped when plateau is reached, it uses Validation loss per default (Default: false)')
    parser.add_argument('--auc', dest='auc', action='store_true',
                        help='Area under the curve scheme for early stopping, if false validation loss will be used - early-stop variable must be set to true (Default: false)')
    parser.add_argument('--adaptive', dest='adaptive',
                        action='store_true', help='Use adaptive network scheme')
    parser.add_argument('--pretrain', dest='pretrain',
                        action='store_true', help='Use pretrained network scheme')
    parser.add_argument('--finetune', dest='finetune',
                        action='store_true', help='Use fintuned network scheme')
    parser.add_argument('--epochs', type=int, dest='epochs',
                        default=100, help='Define number of epochs (Default: 100)')
    parser.add_argument('--lr', type=float, dest='lr', default=0.01,
                        help='Define learning rate (Default: 0.01)')
    parser.add_argument('--momentum', type=float, dest='momentum',
                        default=0.9, help='Define momentum parameter (Default: 0.9)')
    parser.add_argument('--k-gallery', type=int, dest='k_gallery', default=-1,
                        help='k-most similar features (computed with cosine distance) are used for the gallery, the selected features are excluded from training (Default: -1)')
    return parser.parse_args()


def dir_path(path):
    if os.path.isabs(path):
        return path
    raise argparse.ArgumentTypeError(
        f"readable_dir: {path} is not a valid path")


def file_iterable(path):
    for cat_dir in sorted(os.listdir(path)):
        cat_name = os.fsdecode(cat_dir)
        if os.path.isfile(F'{path}{cat_name}'):  # only consider directories
            continue
        for file_name in sorted(os.listdir(F'{path}{cat_name}')):
            yield (cat_name, file_name)


def save_json_file(name, content):
    with open(F"{name}.json", 'w') as fp:
        json.dump(content, fp,  indent=4)


def load_json_file(path):
    f = open(path, "r")
    data = json.loads(f.read())
    f.close()
    return data
