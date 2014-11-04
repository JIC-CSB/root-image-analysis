import argparse
import numpy as np
from skimage.io import use_plugin, imread, imsave
from skimage.morphology import binary_erosion, disk
from skimage.color import rgb2grey

def generate_segmentation_outline(input_file, output_file):
    """Erode and subtract to generate segmentation outline."""
    use_plugin('freeimage')

    segmentation_im = imread(input_file)
    segmentation_grey_im = segmentation_im
    if len(segmentation_im.shape) > 2:
        segmentation_grey_im = rgb2grey(segmentation_im)

    binary_mask_im = (segmentation_grey_im != 0)
    salem = disk(1)
    erosion_mask_im = binary_erosion(binary_mask_im, salem)
    outline_mask_im = np.logical_xor(binary_mask_im, erosion_mask_im)

    if len(segmentation_im.shape) > 2:
        im = np.zeros(segmentation_im.shape, dtype=np.uint8)
        for i in range(3):
            im[:,:,i] = segmentation_im[:,:,i] * outline_mask_im
        imsave(output_file, im)
    else:
        output_im = segmentation_im * outline_mask_im
        imsave(output_file, output_im)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help="Segmented image")
    parser.add_argument('output_file', help="Segmentation outline")
    args = parser.parse_args()
    generate_segmentation_outline(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
