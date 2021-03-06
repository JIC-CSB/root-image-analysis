
import os
import sys
import logging

import numpy as np

from skimage.io import use_plugin, imread, imsave

logger = logging.getLogger('__main__.{}'.format(__name__))

def apply_mask(input_file, mask_file, output_file):
    use_plugin('pil')

    # TODO - shape mismatches

    input_image = imread(input_file)
    mask_image = imread(mask_file)

    logger.info('Input image shape: {}'.format(input_image.shape))
    logger.info('Mask image shape: {}'.format(mask_image.shape))

    xdim, ydim, _ = input_image.shape

    output_image = np.zeros((xdim, ydim), np.uint8)

    mask_xs, mask_ys = np.where(mask_image == 255)

    mask_zs = np.zeros(mask_xs.shape, np.uint8)

    mask_locations = mask_xs, mask_ys, mask_zs
    output_locations = (mask_xs, mask_ys)

    output_image[output_locations] = input_image[mask_locations]

    imsave(output_file, output_image)
    

def main():

    input_file = sys.argv[1]
    mask_file = sys.argv[2]

    output_file = sys.argv[3]

    apply_mask(input_file, mask_file, output_file)

if __name__ == "__main__":
    main()
