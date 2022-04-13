#!/usr/bin/env python3
import data, utilities
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy
import torch
import argparse


vis = {}  # dict stores all layer outputs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Visualizes features map and filters of a provided model')
    parser.add_argument('--model', type=utilities.dir_path,
                        help='Directory where model for visualization is located (absolute dir)')
    parser.add_argument('--d', type=utilities.dir_path,
                        help='Directory where images are stored that will be used for the visualization of the feature maps (absolute dir)')
    parser.add_argument('--fine-tune', dest='fine_tune', action='store_true',
                        help='Define if the whole model is fine-tuned (Default: false)')
    parser.add_argument('--filters', dest='filters', action='store_true',
                        help='Visualize filters of model (Default: false)')
    parser.add_argument('--maps', dest='maps', action='store_true',
                        help='Visualize feature maps of model (Default: false)')
    parser.add_argument('--dm', dest='dm', action='store_true',
                        help='Apply dimensionality reduction (t-sne, pca) on features (Default: false)')
    parser.add_argument('--roc', dest='roc', action='store_true',
                        help='Visualize RoC graph on features (Default: false)')
    parser.add_argument('--hist', dest='hist', action='store_true',
                        help='Visualize magintue graph on features OVR (Default: false)')
    parser.add_argument('--features', type=utilities.dir_path,
                        help='Directory where features are stored (absolute dir)')
    parser.add_argument('--features_test', type=utilities.dir_path,
                        help='Directory where test features are stored (absolute dir)')
    parser.add_argument('--d_test', type=utilities.dir_path,
                        help='Directory where test data are stored (absolute dir)')
    parser.add_argument('--confusion', type=utilities.dir_path,
                        help='Directory where data for confusion matrix is stored (absolute dir)')
    return parser.parse_args()


def visualize_filters(layers):
    for i, l in enumerate(layers):
        w = l.weight
        # construct number of rows and columns for subplot
        size = int(numpy.sqrt(w.shape[0]))
        # check if number of filters can be arranged in subplots
        x = y = int(size) + 1 if size % 1 != 0 else int(size)
        plt.figure(figsize=(20, 17))
        for j, filter in enumerate(w):
            plt.subplot(x, y, j+1)  # use shape of filter to define subplot
            plt.imshow(filter[0, :, :].cpu().detach(), cmap='viridis')
            plt.axis('off')
            plt.savefig(F'Conv_{i}_Filter.png')
        plt.close()


def extractConvLayers(model):
    modules = list(model.children())
    layers = []
    for m in modules:
        if type(m) == torch.nn.Conv2d:
            layers.append(m)
        elif type(m) == torch.nn.Sequential:
            for i in list(m.children()):
                if type(i) == torch.nn.Conv2d:
                    layers.append(i)
    return layers


def hook_fn(module, _, output):
    vis[module] = output


def get_layers(model):
    for _, layer in model._modules.items():
        # recursive call on children of sequential object
        if isinstance(layer, torch.nn.Sequential):
            get_layers(layer)
        else:
            # only register a hook when we do not have a sequential object
            layer.register_forward_hook(hook_fn)


def save_scatter_plot(features, proj, num_categories, name):
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 1, 1)
    index_start = 0
    counter = 0
    for cat, val in features.items():
        # calculate end of category index
        index_end = index_start + val.size(0) - 1
        # extract only values for one category
        proj_cat = proj[index_start:index_end, :]
        ax.scatter(proj_cat[:, 0], proj_cat[:, 1],
                   c=utilities.colors[counter], label=cat, alpha=0.5, marker='+')
        index_start = index_end + 1
        counter += 1

    box = ax.get_position()
    # Shrink current axis's height by 10% on the bottom
    ax.set_position([box.x0, box.y0 + box.height *
                    0.1, box.width, box.height * 0.9])
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, -0.05), ncol=int(num_categories / 8))
    plt.savefig(F'{name}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def normalize_features(features):
    # combine features into one tensor
    vals = torch.cat(tuple(features.values()), dim=0)
    norm = torch.linalg.norm(vals, dim=0)  # compute norm over the columns
    tol = 1e-12  # tolerance

    # check if computed norm is greater than tolerance to prevent divsion by zero
    norm[norm < tol] = tol

    # apply normalization
    return {k: normalize(v, norm) for k, v in features.items()}


def normalize(features, norm):
    # use same norm from training features
    return torch.div(features, norm)


def main():
    parsed_args = parse_arguments()
    features = torch.load(parsed_args.features)
    features_test = torch.load(parsed_args.features_test)

    if parsed_args.dm:

        # get feature dimension (retrieved from first element)
        f_dim = (list(features.values())[0]).size(1)
        num_categories = len(features.keys())  # get number of categories
        # init embedding with calculated f dimension
        embeddings = torch.zeros((0, f_dim), dtype=torch.float32)
        for f in features.values():
            # stack all features from every class into one embeddings tensor
            embeddings = torch.cat((embeddings, f))

    # invoke t-SNE on stacked feature embedding
    tsne = TSNE(n_components=2, perplexity=40, init='pca',
                learning_rate='auto', verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)

    # visualize t-sne with coloring of correct class
    save_scatter_plot(features, tsne_proj, num_categories, 'tsne')

    # invoke pca algo
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(embeddings)
    print(F'Variance ratio: {pca.explained_variance_ratio_}')

    # visualize pca with coloring of correct class
    save_scatter_plot(features, pca_proj, num_categories, 'pca')

    if parsed_args.roc:
        utilities.perform_roc(features, features_test)

    if parsed_args.hist:
        print(features)
        utilities.save_feature_magnitude_hist(features)

    if parsed_args.confusion is not None:
        res_data = utilities.load_json_file(parsed_args.confusion)

        # only test on 70 images per class at the moment
        utilities.save_confusion_matrix(res_data["5"])

    if parsed_args.filters or parsed_args.maps:
        # load test data
        test_data = data.CustomImageDataset(
            'data.csv', parsed_args.d, utilities.test_transforms())
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data, batch_size=1, shuffle=False)

        # define and load existing model
        model = utilities.define_model(test_data, parsed_args.fine_tune)
        model.load_state_dict(torch.load(parsed_args.model))
        model.eval()

    # visualize convolutional layers with input images
    if parsed_args.filters:
        layers = extractConvLayers(model)
        visualize_filters(layers)

    if parsed_args.maps:
        model.cpu()
        get_layers(model)
        with torch.no_grad():
            for img, label, name, _ in test_loader:
                print(F"Save Feature maps for category {label} -> {name[0]}")
                _ = model(img)
                break

        for mod, output in vis.items():
            # remove batch dimension
            d = output.squeeze()

            # construct number of rows and columns for subplot
            size = int(numpy.sqrt(d.size(0)))
            # check if number of filters can be arranged in subplots
            x = y = int(size) + 1 if size % 1 != 0 else int(size)
            plt.figure(figsize=(20, 17))
            for j, activation in enumerate(d):
                plt.subplot(x, y, j+1)  # use shape of filter to define subplot
                plt.imshow(activation, cmap='viridis')
                plt.axis('off')
                plt.savefig(F'{mod._get_name()}_Activation.png')
            plt.close()
            break


if __name__ == "__main__":
    main()
