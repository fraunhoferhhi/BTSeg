# flake8: noqa: E201, E202

from collections import namedtuple

import matplotlib.patches as mpatches
import numpy as np

# partly taken from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        "id",
        "trainId",  # Feel free to modify these IDs as suitable for your method.
        "color",  # The color of this label
    ],
)


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# fmt: off
CityscapesLabels = [
    #       name                        id   trainId
    Label(  'unlabeled'            ,  0 ,       255 , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,       255 , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,       255 , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,       255 , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,       255 , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,       255 , (  0,  0,  0) ),
    Label(  'ground'               ,  6 ,       255 , (  0,  0,  0) ),
    Label(  'road'                 ,  7 ,        0 , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , (244, 35,232) ),
    Label(  'parking'              ,  9 ,       255 , (  0,  0,  0) ),
    Label(  'rail track'           , 10 ,       255 , (  0,  0,  0) ),
    Label(  'building'             , 11 ,        2 , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , (190,153,153) ),
    Label(  'guard rail'           , 14 ,       255 , (  0,  0,  0) ),
    Label(  'bridge'               , 15 ,       255 , (  0,  0,  0) ),
    Label(  'tunnel'               , 16 ,       255 , (  0,  0,  0) ),
    Label(  'pole'                 , 17 ,        5 , (153,153,153) ),
    Label(  'polegroup'            , 18 ,       255 , (  0,  0,  0) ),
    Label(  'traffic light'        , 19 ,        6 , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,       255 , (  0,  0,  0) ),
    Label(  'trailer'              , 30 ,       255 , (  0,  0,  0) ),
    Label(  'train'                , 31 ,       16 , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       255 , (  0,  0,  0) ),
]
# fmt: on

# Create dictionaries for a fast lookup

# name to label object
id2label = {label.id: label for label in CityscapesLabels}
trainId2label = {label.trainId: label for label in reversed(CityscapesLabels)}


def get_label_train_classes():
    """Returns a list of all train classes.

    Returns:
        list: A list of all train classes.
    """
    return [label.name for label in CityscapesLabels if label.trainId != 255] + ["unlabelled"]


def get_id_to_label_mapping():
    """Returns a dictionary that maps the ids to the label objects.

    Returns:
        dict: A dictionary that maps the ids to the label objects.
    """
    return id2label


def get_train_id_to_name_mapping():
    """Returns a dictionary that maps the trainIds to the label objects.

    Returns:
        dict: A dictionary that maps the trainIds to the label objects.
    """

    return {label.trainId: label.name for label in reversed(CityscapesLabels)}


def get_matplotlib_patches_for_cityscapes_color_legend():
    """Returns a list of matplotlib.patches.Patch objects that can be used to create a legend for
    the cityscapes color scheme.

    Usage:
        patches = get_matplotlib_patches_for_cityscapes_color_legend()
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    """

    colormap = {
        label.name: np.array(list(label.color) + [255]) / 255.0
        for label in CityscapesLabels
        if not label.trainId == 255
    }
    # add a black color for the unlabeled class
    colormap["unlabeled"] = np.array([0, 0, 0, 255]) / 255.0

    patches = [mpatches.Patch(color=color, label=label) for label, color in colormap.items()]

    return patches


def get_colorized_image_from_predictions(pred: np.array):
    h = pred.shape[0]
    w = pred.shape[1]

    pred_image = np.ones((h, w, 3)) * 255.0

    for label in CityscapesLabels:
        if label.trainId == 255:
            continue

        # print(f"Train ID: {label.trainId} Color({label.color})")

        pred_image[pred == label.trainId] = label.color

    # unlabeled
    pred_image[pred == 19] = (0, 0, 0)

    pred_image = pred_image / 255.0

    return pred_image
