"""Sum the total area (in pixels) covered by a segmentation"""

import argparse

import numpy as np
from skimage.io import use_plugin, imread

use_plugin('freeimage')

def sum_segmented_area(segmentation_file):

    im_array = imread(segmentation_file)

    area = len(np.where(im_array != 0)[0])

    return area

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('segmentation_file', help="File containing segmentation")

    args = parser.parse_args()

    print sum_segmented_area(args.segmentation_file)

if __name__ == '__main__':
    main()
