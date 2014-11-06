import os
import argparse

import cv2

def remove_border_segmentations(input_file, output_file):
    """Remove segments that touch the image border."""
    im = cv2.imread(input_file, -1)
    xdim, ydim = im.shape
    print('xdim {}; ydim {}'.format(xdim, ydim))

    # Identify any segments that touch the image border.
    border_seg_ids = set()
    border_seg_ids.update(set(im[0,:]))
    border_seg_ids.update(set(im[:,0]))
    border_seg_ids.update(set(im[xdim-1,:]))
    border_seg_ids.update(set(im[:,ydim-1]))

    # Remove those segments.
    for bs_id in border_seg_ids:
        mask = ( im != bs_id )
        im = im * mask

    cv2.imwrite(output_file, im)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='Input image file')
    parser.add_argument('output_file', help='Output image file')
    args = parser.parse_args()
    if not os.path.isfile(args.input_file):
        parser.error('No such file: {}'.format(args.input_file))

    remove_border_segmentations(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
