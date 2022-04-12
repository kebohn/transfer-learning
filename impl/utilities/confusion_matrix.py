import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy
import json


def save_confusion_matrix(res_data):

    # f = open("/local/scratch/bohn/results/indoor/classes.json", "r")
    # labels_str = json.loads(f.read())
    # f.close()

    # create confusion matrix
    labels = list(numpy.unique(numpy.array(res_data["labels"])))
    confusion = confusion_matrix(
        res_data["labels"], res_data["predictions"], labels=labels)

    # transpose matrix because we want the rows to be the predicted class
    confusion = confusion.T

    # compute rounded accuracy values per class
    acc_row = numpy.diag(confusion) / confusion.sum(axis=0)
    acc_row = numpy.around(acc_row * 100, decimals=0)

    # append accuracy per class row at the end of matrix
    confusion = numpy.vstack([confusion, acc_row]).astype(int)
    y_labels = labels.copy()
    y_labels.append("%")

    fig, ax = plt.subplots(figsize=(20, 20))

    # mask the accuracy / bottom row
    confusion_bottom = confusion.copy()
    masked_array = numpy.ones(confusion.shape, dtype=bool)
    false_list = numpy.zeros((1, confusion.shape[1]), dtype=bool)
    masked_array[-1, :] = false_list
    mask_confusion_bottom = numpy.ma.MaskedArray(
        confusion_bottom, mask=masked_array)

    # mask diagonal matrix
    confusion_diag = confusion.copy()
    numpy.fill_diagonal(confusion_diag, acc_row)
    masked_array = numpy.ones(confusion.shape, dtype=bool)
    false_list = numpy.zeros((1, confusion.shape[1]), dtype=bool)
    numpy.fill_diagonal(masked_array, false_list)
    mask_confusion_diag = numpy.ma.MaskedArray(
        confusion_diag, mask=masked_array)

    im = ax.imshow(confusion, cmap='jet', vmin=0,
                   vmax=numpy.amax(confusion[:-1, :]))
    im_bottom = ax.imshow(
        confusion_bottom, cmap=mpl.colors.ListedColormap(['white']))
    im_diag = ax.imshow(confusion_diag, cmap='jet',
                        vmin=0, vmax=numpy.amax(acc_row))

    im_bottom.set_data(mask_confusion_bottom)
    im_diag.set_data(mask_confusion_diag)

    # show ticks and labels on both axes
    ax.set_xticks(numpy.arange(len(labels)))
    ax.set_yticks(numpy.arange(len(y_labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Estimated')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90,
             ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        color = "w" if i != confusion.shape[0] - 1 else "black"
        for j in range(len(labels)):
            if confusion[i, j] != 0:  # do not show 0 values
                ax.text(j, i, confusion[i, j],
                        ha="center", va="center", color=color)

    ax.set_title("Confusion matrix Indoor")
    fig.tight_layout()
    plt.tick_params(bottom=False)
    plt.tick_params(left=False)
    plt.savefig("confusion_matrix.png")
    plt.close()
