#!/usr/bin/env python3
import copy
from modulefinder import STORE_OPS
import torch
import numpy
import data, models, utilities


def main():
    parsed_args = utilities.parse_arguments()
    gallery_loader = {}
    res = {}

    print("Prepare datasets...")
    valid_data = data.CustomImageDataset(
        'data.csv',
        parsed_args.d_valid,
        utilities.test_transforms()
    )
    test_data = data.CustomImageDataset(
        'data.csv',
        parsed_args.d_test,
        utilities.test_transforms()
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=10,
        shuffle=True,
        num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=10 if parsed_args.adaptive else 1,
        shuffle=False,
        num_workers=8
    )

    # load gallery data - gallery.csv must be generated beforehand
    if parsed_args.k_gallery:

        gallery_data = data.CustomImageDataset(
            'gallery.csv',
            parsed_args.d,
            utilities.test_transforms()
        )

        gallery_loader = torch.utils.data.DataLoader(
            dataset=gallery_data,
            batch_size=len(gallery_data),
            shuffle=False,
            num_workers=8
        )

    # load different models
    print("Load models...")
    loaded_model = models.get_pretrained_model(parsed_args.model_type)
    extraction_model = models.FEModel(
        model=loaded_model,
        model_type=parsed_args.model_type,
        device=utilities.get_device())

    # extract validation, test and gallery features for adaptive model
    if parsed_args.adaptive:
        last_layer = models.get_last_layer(loaded_model, parsed_args.model_type)
        adaptive_model = models.AdaptiveModel(
            fc1_in=last_layer.in_features,
            fc1_out=parsed_args.fc1_out,
            feature_size=parsed_args.feature_size,
            category_size=test_data.get_categories()
        )
        valid_features = utilities.extract(extraction_model, valid_loader)
        test_features = utilities.extract(extraction_model, test_loader)
        if parsed_args.k_gallery:
            gallery_features = utilities.extract(extraction_model, gallery_loader)

    # define log steps starting from 1 till max_size
    steps = numpy.logspace(0.1, numpy.log10(parsed_args.max_size), num=parsed_args.num_steps, endpoint=True, dtype=int)
    steps = numpy.unique(steps) # remove duplicates - especially occurs at the beginning of list
    steps[-1] = parsed_args.max_size # make sure we do not have a rounding error

    # iterate over steps
    for current_size in steps:
        print(F'Using {current_size} images per category...')

        # clear cache after each iteration
        torch.cuda.empty_cache()

        # load training data
        train_data = data.CustomImageDataset(
            'training.csv' if parsed_args.k_gallery else 'data.csv',
            parsed_args.d,
            utilities.test_transforms() if parsed_args.pretrain else utilities.train_transforms(),
            current_size
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=10,
            shuffle=True,
            num_workers=8
        )

        # adaptive model case
        if parsed_args.adaptive:
            # copy the original model
            model = copy.deepcopy(adaptive_model)

            # load model from disk
            if parsed_args.load is not None:
                path = utilities.find_file_path(parsed_args, current_size)
                model.load_state_dict(torch.load(path))
                model.to(utilities.get_device())  # save to GPU
                train_features_loader, _ = utilities.prepare_features_adaptive_training(extraction_model, train_loader, valid_features)

            # train model from scratch
            else:
                model.to(utilities.get_device())  # save to GPU

                # train data with current size of samples per category
                train_features_loader, _ = utilities.train(
                    pre_trained_model=extraction_model,
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    valid_features=valid_features,
                    params=parsed_args,
                    current_size=current_size
                )

            # use Feature Extraction Model to prepare input data for adaptive model
            # normalize test data
            test_features_norm = extraction_model.normalize_test(test_features)

            # handle test features like a dataset
            test_data_adaptive = data.FeatureDataset(test_features_norm)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_data_adaptive,
                batch_size=1,
                shuffle=False
            )
            if parsed_args.k_gallery:
                # normalize permanent gallery data
                gallery_features_norm = extraction_model.normalize_test(gallery_features)
                # handle permanent gallery features like a dataset
                gallery_data_adaptive = data.FeatureDataset(gallery_features_norm)
                gallery_loader = torch.utils.data.DataLoader(
                    dataset=gallery_data_adaptive,
                    batch_size=len(gallery_data_adaptive),
                    shuffle=False,
                    num_workers=8
                )

        # fine-tuned model case
        elif parsed_args.finetune:
            # copy the original model
            model = copy.deepcopy(loaded_model)

            # change last layer out neurons to respective number of classes from the dataset
            models.update_last_layer(model, parsed_args.model_type, test_data.get_categories())

            # load model from disk
            if parsed_args.load is not None:
                path = utilities.find_file_path(parsed_args, current_size)
                model.load_state_dict(torch.load(path))
                model.to(utilities.get_device())  # save to GPU
                train_features_loader = copy.deepcopy(train_loader)
            
            # train model from scratch
            else:
                model.to(utilities.get_device())  # save to GPU

                # train data with current size of samples per category
                train_features_loader, _ = utilities.train(
                    pre_trained_model=extraction_model,
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    params=parsed_args,
                    current_size=current_size
                )

        # extract features from model and use this with another specified metric to predict the categories
        if parsed_args.extract:
            if parsed_args.adaptive or parsed_args.finetune:

                # define trained adaptive extraction model
                learned_extraction_model = models.FEModel(
                    model=model,
                    model_type=parsed_args.model_type if parsed_args.finetune else 'adaptive',
                    device=utilities.get_device()
                )

                # load stores features from disk
                if parsed_args.load_features:
                    tr_features, ga_features = utilities.load_features(
                    current_size=current_size,
                    params=parsed_args)
                else:
                    # extract features from trained model
                    tr_features, ga_features = utilities.extract_all_features(
                        model=learned_extraction_model,
                        tr_loader=train_features_loader,
                        te_loader=test_loader,
                        ga_loader=gallery_loader,
                        current_size=current_size,
                        params=parsed_args
                    )

            else:
                # load stores features from disk
                if parsed_args.load_features:
                    tr_features, ga_features = utilities.load_features(
                    current_size=current_size,
                    params=parsed_args)
                else:
                    # extract features from pretrained model
                    tr_features, ga_features = utilities.extract_all_features(
                        model=extraction_model,
                        tr_loader=train_loader,
                        te_loader=test_loader,
                        ga_loader=gallery_loader,
                        current_size=current_size,
                        params=parsed_args
                    )

            # run prediction
            res[current_size] = utilities.predict(
                model=extraction_model if parsed_args.pretrain else learned_extraction_model,
                params=parsed_args,
                features=ga_features if parsed_args.k_gallery else tr_features,
                test_loader=test_loader,
            )

        # use the model to classify the images
        else:
            res[current_size] = utilities.test(model, test_loader)

    utilities.save_json_file(F'{parsed_args.results}res', res)
    utilities.save_training_size_plot(parsed_args.results, res)


if __name__ == "__main__":
    main()
