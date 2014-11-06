"""Sum pixel areas convered by a directory of segmentations"""

import os
import argparse

from sum_segmentation_area import sum_segmented_area

def sum_segmentation_dir(segmentation_dir):
    all_seg_files = os.listdir(segmentation_dir)

    full_file_paths = [os.path.join(segmentation_dir, sf) for sf in all_seg_files]

    return sum(map(sum_segmented_area, full_file_paths))

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('segmentation_dir', help='Directory containing segmentations.')

    args = parser.parse_args()

    print sum_segmentation_dir(args.segmentation_dir)

if __name__ == '__main__':
    main()