"""
Script to create a mask for the main object in an image.

For example the outline of a root.
"""

import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import scipy.ndimage
import skimage.io
import skimage.color
import skimage.filter
import skimage.morphology
import time

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
#logger.setLevel(logging.DEBUG)

def imshow(image):
    """Display the image.
    
    For debugging purposes.
    """
    if logger.isEnabledFor(logging.DEBUG):
        skimage.io.imshow(image)
        plt.show()

def inverse(image):
    "Return the negative image."
    return 1 - image

def get_object_mask_image(img, min_size, dilate_size):
    "Return binary image of the object mask."
    logger.info('gaussian...')
    img = skimage.filter.gaussian_filter(img, 1)
    imshow(img)

    logger.info('threshold...')
    threshold = skimage.filter.threshold_otsu(img)
    img = img <= threshold
    imshow(img)

    logger.info('inverse...')
    img = inverse(img)
    imshow(img)

    logger.info('fill holes...')
    img = scipy.ndimage.binary_fill_holes(img)
    imshow(img)

    logger.info('remove small objects (<{})...'.format(min_size))
    img = skimage.morphology.remove_small_objects(img, min_size=min_size)
    imshow(img)

    logger.info('dilate {}...'.format(dilate_size))
    salem = skimage.morphology.disk(dilate_size)
    img = skimage.morphology.binary_dilation(img, salem)
    imshow(img)

    logger.info('convex hull...')
    img = skimage.morphology.convex_hull_image(img)
    imshow(img)

    logger.info('erode...')
    salem = skimage.morphology.disk(50)
    img = skimage.morphology.binary_erosion(img, salem)
    imshow(img)

    return img

def get_grey_image(img):
    """Return 2D array image.

    Caveat: If the input image is 3D it assumed to be an rgb image!

    Is there a better way of dealing with different types of images and
    converting them to 2D arrays?
    """
    if img.ndim == 3:
        return skimage.color.rgb2grey(img)
    return img

def get_color_image(img):
    """Return 2D array in range 0 to 255.
    
    Assumes the input is binary (0,1).
    """
    return img * 255

def generate_object_mask(input_fn, output_fn, min_size, dilate_size):
    skimage.io.use_plugin('pil')
    img = skimage.io.imread(input_fn)
    img = get_grey_image(img)
    img = get_object_mask_image(img, min_size, dilate_size)
    img = get_color_image(img)
    skimage.io.imsave(output_fn, img)


def main(input_fn, output_fn, min_size, dilate_size):
    "The control logic of the script."
    img = skimage.io.imread(input_fn)
    img = get_grey_image(img)
    img = get_object_mask_image(img, min_size, dilate_size)
    img = get_color_image(img)
    skimage.io.imsave(output_fn, img)

if __name__ == '__main__':
    "Parse the command line arguments."
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='path to input file')
    parser.add_argument('output_file', help='output file name')
    parser.add_argument('--min_size',
        default=1000,
        type=int,
        help='objects below this size will be filtered out')
    parser.add_argument('--dilate_size',
        default=6,
        type=int,
        help='how much to dilate after having removed small objects')
    args = parser.parse_args()
    if not os.path.isfile(args.input_file):
        parser.error('No such file: {}'.format(args.input_file))
    main(args.input_file, args.output_file, args.min_size, args.dilate_size)
