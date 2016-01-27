"""Generate partial 3D reconstruction from series of 2D slices"""


import re
import os
import argparse

import random
import numpy as np
from skimage.io import use_plugin, imread, imsave
import matplotlib.pyplot as plt

from reconstructor import Reconstruction, load_segmentation_maps
from sum_segmentation_dir import sum_segmentation_dir

import logging
logger = logging.getLogger('__main__.{}'.format(__name__))

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def shades_of_jop():
    """Return a pretty colour."""
    c1 = random.randint(127, 255) 
    c2 = random.randint(0, 127) 
    c3 = random.randint(0, 255) 

    return tuple(random.sample([c1, c2, c3], 3))

SHADES_OF_JOP_USED = set()
def shades_of_jop_unique():
    """Return a unique pretty colour."""
    jop = shades_of_jop()
    if jop in SHADES_OF_JOP_USED:
        jop = shades_of_jop_unique()
    else:
        SHADES_OF_JOP_USED.add(jop)
    return jop

def get_mask_output_fpaths(seg_dir, out_dir, start_z, end_z):
    """Return list of reconstruction mask output file paths."""
    fnames = os.listdir(seg_dir)
    fnames = sorted_nicely(fnames)
    fpaths = [os.path.join(out_dir, fn) for fn in fnames]
    return fpaths[start_z:end_z+1]  # Plus one is intentional; end_z goes to z-1

def generate_reconstruction_mask(out_fpaths, xdim, ydim, rcells, start_z, end_z):
    """Generate reconstruction mask."""
    use_plugin('freeimage')

    das = {z: np.zeros((xdim, ydim, 3), dtype=np.uint8)
           for z in range(start_z, end_z+1)}  # Plus one is intentional; end goes to z-1

    for rcell in rcells:
        c = shades_of_jop_unique()
        for z, cellslice in rcell.slice_dict.items():
            das[z][cellslice.coord_list] = c

    for out_fn, z in zip(out_fpaths, range(start_z, end_z+1)):
#       out_fn = os.path.join(out_dir, "da%d.tif" % z)
        imsave(out_fn, das[z])
        

def load_intensity_data(intensity_dir):
    
    image_files = os.listdir(intensity_dir)

    image_files = sorted_nicely(image_files)

    idata = [imread(os.path.join(intensity_dir, im_file))
             for im_file in image_files]

    return idata

def generate_heatmap(seg_dir, measure_dir, out_dir, start_z, end_z):
    logger.info('Segmentation dir: {}'.format(seg_dir))
    logger.info('Measurement dir: {}'.format(measure_dir))
    logger.info('Output dir: {}'.format(out_dir))
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

    output_filename = "heatmap.png"
    full_output_path = os.path.join(out_dir, output_filename)
    heatmap_array = np.zeros((xdim, ydim), 3), dtype=np.uint8)

    selected_z_plane = 7
    imax = 4
    imin = 0.9

    for rcell in rcells:
        if selected_z_plane in rcell.slice_dict:
            intensity = rcell.measure_mean_intensity(idata)
            gval = 255 * (intensity - imin) / (imax - imin)
            gval = min(255, gval)
            heatmap_array[rcell.slice_dict[selected_z_plane].coord_list] = [255-gval, gval, 0]

    imsave(full_output_path, heatmap_array)


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

    generate_heatmap(args.seg_dir, args.measure_dir, args.out_dir,
                     args.z_start, args.z_end)

    

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    main()
