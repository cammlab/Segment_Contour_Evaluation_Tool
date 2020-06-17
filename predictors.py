""" Defines a plug-in module to add predictors. Each predictor has a name that should be added to
    the options dictionary. This dictionary links the name with the predictor function name.

    The predictor function can be added also in this module (see the baseline predictor as example)

    The predictor function takes a class image as an input and outputs a labeled image as the output

"""
import numpy as np
import skimage.io
import skimage.morphology


class Predictor:
    """ Provides access to any number of predictor functions using a factory method """
    def __init__(self, resources):
        self.resources = resources

        # Add predictor names to this structure and add predictor code to the bottom of the file
        self.options = {
            "baseline": baseline_predictor
        }

    def predict(self):
        # Check if the user-entered predictor is one of the defined options
        if self.resources.predictor not in self.options.keys():
            raise ValueError(self.resources.predictor)

        # Invoke the selected predictor with the class file (path) as its argument
        return self.options[self.resources.predictor](self.resources.class_file)


def baseline_predictor(class_image_path):
    """ Initial implementation of a method that converts a class map into labeled segments"""
    # Class image has 3 planes for background, interior, and boundaries. Each plan show the
    # probability that the pixel belongs to the class
    class_image_data = skimage.io.imread(class_image_path)

    # Each pixel in 'pred' gets the plane index (0, 1, of 2) for which the plane value (prob) is max
    # Hence, pred is an image whose pixels have values 0 (background), 1 (interior) or 2 (boundary)
    pred = np.argmax(class_image_data, -1)

    cell_min_size = 25
    cell_label = 1  # This value corresponds to the interior class

    # 'cell' is an image that contains only interior
    cell = (pred == cell_label)

    # Remove small holes and small objects from 'cell'
    cell = skimage.morphology.remove_small_holes(cell, area_threshold=cell_min_size)
    cell = skimage.morphology.remove_small_objects(cell, min_size=cell_min_size)

    # Convert the 'cell' image into an image that contains labels (one label for each segmented nucleus)
    [label, _] = skimage.morphology.label(cell, return_num=True)
    return label
