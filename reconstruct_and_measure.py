"""Generate partial 3D reconstruction from series of 2D slices"""


import re
import os
import argparse

import numpy as np
from skimage.io import use_plugin, imread, imsave
import matplotlib.pyplot as plt

from reconstructor import Reconstruction, load_segmentation_maps

import logging
logger = logging.getLogger('__main__.{}'.format(__name__))

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def load_intensity_data(intensity_dir):
    
    image_files = os.listdir(intensity_dir)

    image_files = sorted_nicely(image_files)

    idata = [imread(os.path.join(intensity_dir, im_file))
             for im_file in image_files]

    return idata

def reconstruct_and_measure(seg_dir, measure_dir,
                            out_dir, results_file,
                            start_z, end_z):
    logger.info('Segmentation dir: {}'.format(seg_dir))
    logger.info('Measurement dir: {}'.format(measure_dir))
    logger.info('Output dir: {}'.format(out_dir))
    logger.info('Results file: {}'.format(results_file))
    use_plugin('freeimage')

    smaps = load_segmentation_maps(seg_dir)
    idata = load_intensity_data(measure_dir)

    xdim, ydim = idata[0].shape

    if start_z is None:
        start_z = 0
    if end_z is None:
        end_z = len(smaps)-1  # Minus 1 is intentional; need to be able to extend one more z-stack
    logger.info('Start z: {:d}'.format(start_z))
    logger.info('End z: {:d}'.format(end_z))

    r = Reconstruction(smaps, start=start_z) 
    logger.debug('Reconstruction instance: {}'.format(r))

    for z in range(start_z, end_z):
        r.extend(z)

    rcells = r.cells_larger_then(3)

    with open(results_file, "w") as f:
        f.write('mean_intensity,quartile_intensity,best_intensity,best_z,x,y,z,volume,zext')
        for rcell in rcells:
            x, y, z = rcell.centroid
            mean_intensity = rcell.measure_mean_intensity(idata)
            quartile_intensity = rcell.measure_quartile_intensity(idata)
            best_intensity, best_z = rcell.measure_best_slice(idata)
            volume = rcell.pixel_area
            zext = rcell.z_extent

            f.write("{},{},{},{},{},{},{},{},{}\n".format(
                mean_intensity,
                quartile_intensity,
                best_intensity,
                best_z,
                x, y, z,
                volume,
                zext))

def main():
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seg_dir', help="Path to directory containing segmented images")
    parser.add_argument('measure_dir', help="Path to directory containing intensity images")
    parser.add_argument('out_dir', help="Path to output directory")
    parser.add_argument('results_file', help="Filename to which results should be output")
    parser.add_argument('--z_start', help="First z-stack",
                        default=None, type=int)
    parser.add_argument('--z_end', help="Last z-stack",
                        default=None, type=int)

    args = parser.parse_args()

    recons = reconstruct_and_measure(args.seg_dir, args.measure_dir,
                                     args.out_dir, args.results_file,
                                     args.z_start, args.z_end)

    

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    main()
