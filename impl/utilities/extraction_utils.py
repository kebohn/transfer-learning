from utilities.cuda import get_device
import torch
import matplotlib.pyplot as plt
import numpy
import collections
import models
import data


def softmax(x):
    return numpy.exp(x) / sum(numpy.exp(x))


def prepare_features_adaptive_training(pre_trained_model, train_loader, valid_features):

    # extract training features from training data
    train_features = extract(pre_trained_model, train_loader)

    # normalize training features
    train_features_norm = pre_trained_model.normalize_train(train_features)

    # handle trainig features like a dataset
    train_features_data = data.FeatureDataset(train_features_norm)
    train_features_loader = torch.utils.data.DataLoader(
        dataset=train_features_data, batch_size=10, shuffle=True)

    # normalize validation features according train normalization
    valid_features_norm = pre_trained_model.normalize_test(valid_features)

    # handle validation features like a dataset
    valid_features_data = data.FeatureDataset(valid_features_norm)
    valid_feature_loader = torch.utils.data.DataLoader(
        dataset=valid_features_data, batch_size=10, shuffle=False)

    return train_features_loader, valid_feature_loader


def extract_all_features(model, tr_loader, te_loader, ga_loader, current_size, params):
    ga_features = {}
    tr_features = extract(model, tr_loader)
    te_features = extract(model, te_loader)
    torch.save(tr_features, F'{params.results}features_train_size_{current_size}.pt')
    torch.save(te_features, F'{params.results}features_test_size_{current_size}.pt')
    if params.k_gallery:
        ga_features = extract(model, ga_loader)
        torch.save(ga_features, F'{params.results}features_gallery_size_{current_size}.pt')
    return tr_features, ga_features


def load_features(current_size, params):
    ga_features = {}
    tr_features = torch.load(F'{params.load_features}features_train_size_{current_size}.pt', map_location=get_device())
    if params.k_gallery:
        ga_features = torch.load(F'{params.load_features}features_gallery_size_{current_size}.pt', map_location=get_device())
    return tr_features, ga_features


def extract(model, train_loader):
    print("Extract features...")
    res = {}
    # iterate over training data
    for values, _, names, _ in train_loader:
        features = model.extract(values)  # extract features for whole batch
        cat_set = set(names)
        names_arr = numpy.array(names)
        for category in cat_set:  # iterate over all distinctive categories in the batch
            # find indices from the same category
            indices = numpy.argwhere(names_arr == category).flatten()
            features = features.unsqueeze(dim=0) if len(
                features.size()) == 1 else features  # make sure we have a 2D tensor
            cat_features = torch.index_select(features, 0, torch.from_numpy(indices).to(
                get_device()))  # retrieve only features from correct category
            if category in res.keys():  # check if we already have some features
                # add new features to existing ones
                res[category] = torch.cat((res[category], cat_features), dim=0)
            else:
                res[category] = cat_features  # add new features
    return dict(sorted(res.items()))


def save_training_size_plot(res_dir, res):
    plt.figure()
    plt.plot(list(res.keys()), [obj["total_acc"] for obj in res.values()])
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.savefig(F'{res_dir}total_acc.jpg')
    plt.close()


def class_acc(values, category):
    print(
        F"Classified {values[0]:2} of {values[1]} images ({100 * values[0] / values[1]:6.2f}%) with {category}"
    )
    return values[0] / values[1]


def total_acc(categories, class_accs, params):
    if params.unbalanced:
        acc_mean = numpy.mean(class_accs)
        print(F"\nAccuracy: {acc_mean * 100:6.2f}%")
        return acc_mean
    total = numpy.sum(list(categories.values()), axis=0)
    print(
        F"\nAccuracy: {total[0]:3} of {total[1]} images ({100 * total[0] / total[1]:6.2f}%)")
    return total[0] / total[1]


def predict(model, params, features=[], test_loader=[]):
    print("Scoring...")
    # store number of correct identifictions and total number of identifications per category
    categories = collections.defaultdict(lambda: [0, 0])
    category = ""
    distances = []
    labels = []
    res = {}
    res["cat_acc"] = []
    res["categories"] = []
    res["labels"] = []
    res["predictions"] = []

    if params.knn:
        # variable with maximum number of features for one category
        max_number_features = max([len(f) for f in features.values()])
        # preserve all distances here
        distances = numpy.zeros((len(features), max_number_features))

    if params.svm:
        svm_model = models.SVMModel(device="not used here")
        y_train = []
        x_train = []
        features_norm = model.normalize_train(features)
        for key, val in features_norm.items():
            y_train.extend(numpy.repeat(key, val.size()[0]))
            tmp = numpy.split(val.detach().cpu().numpy(), val.size()[0])
            x_train.extend([i.flatten() for i in tmp])

        svm_model.fit(x_train, y_train)

    for test_data, _, test_name, _ in test_loader:

        # convert tuple to string
        test_name = ''.join(test_name)

        # add test label to res array
        res["labels"].append(test_name)

        # extract test feature from model
        x_test = model.extract(test_data)

        if params.svm:
            # normalize test data with norm from training data
            x_test_norm = model.normalize(x_test)
            y_test = svm_model.predict(
                x_test_norm.detach().cpu().reshape(1, -1))
        else:
            y_test, _ = model.predict(
                x_test, features, distances, labels, params)

        # add test prediction to res array
        res["predictions"].append(y_test)

        # we only increase when category has been correctly identified
        categories[test_name][0] += y_test == test_name
        # always increase after each iteration s.t. we have the total number
        categories[test_name][1] += 1

        if category != test_name and category:
            # print rates for the current category
            res["cat_acc"].append(class_acc(categories[category], category))
            res["categories"].append(category)
        category = test_name

    res["cat_acc"].append(class_acc(categories[category],category))  # print last category
    res["categories"].append(category)
    res["total_acc"] = total_acc(categories, res["cat_acc"], params)  # print total accuracy

    return res
