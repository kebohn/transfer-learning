import os
import argparse
import json
import glob
import utilities


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Script for different transfer learning approaches')
    parser.add_argument('--d', type=utilities.dir_path,
                        help='Directory where files are located (absolute dir)')
    parser.add_argument('--d-test', type=utilities.dir_path, dest='d_test',
                        help='Directory where test files are located (absolute dir)')
    parser.add_argument('--d-valid', type=utilities.dir_path, dest='d_valid',
                        help='Directory where validation files are located (absolute dir)')
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
                        help='Define step with which training set should be increased (Default: k=5)')
    parser.add_argument('--max-size', type=int, dest='max_size', default=5,
                        help='Define maximum samples per class (Default: k=5)')
    parser.add_argument('--unbalanced', dest='unbalanced', action='store_true',
                        help='Define if dataset is unbalanced (Default: false)')
    parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                        help='''Define if training should be stopped when plateau is reached,
                        it uses Validation loss per default (Default: false)''')
    parser.add_argument('--auc', dest='auc', action='store_true',
                        help='''Area under the curve scheme for early stopping, if false validation loss will be used
                        - early-stop variable must be set to true (Default: false)''')
    parser.add_argument('--threshold', type=int, dest='threshold', default=10,
                        help='''Threshold counter which defines how often validation loss or auc can be exceeded
                        before model training is automatically stopped - early-stop variable must be set to true to take an effect (Default: 10)''')
    parser.add_argument('--decay', dest='decay', action='store_true',
                        help='''Enable Learning Rate Decay when validation loss / auc is not changing (Default: false)''')
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
    parser.add_argument('--k-gallery', dest='k_gallery', action='store_true',
                        help='''k-most similar features (computed with cosine distance) are used for the gallery,
                        the selected features are excluded from training (Default: -1)''')
    parser.add_argument('--model-type', type=str, dest='model_type', default="resnet50",
                        help='''Defines the model type that will be used for the pre-trained model, choose between
                        following parameters: [resnet50, alexnet, vgg16, vgg19, densenet] (Default: resnet50)''')
    parser.add_argument('--load', type=utilities.dir_path, dest='load',
                    help='''Directory where model and features are stored''')
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
    with open(path, "r") as file:
        data = json.loads(file.read())
    return data


def load_multiple_json_files(path):
    data = {}
    for file in os.listdir(path):
        if file.endswith('.json'):
            with open(F"{path}{file}", "r") as f:
                data[file] = json.loads(f.read())
    return data


def find_file_path(params, current_size):
    return glob.glob(F"{params.load}model_size_{current_size}_lr_{params.lr}_*.pth")[0] # return single model path with wildcard at the end
