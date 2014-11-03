"""Generate partial 3D reconstruction from series of 2D slices"""


import re
import os
import sys
import random
import argparse
from pprint import pprint

import numpy as np
from skimage.io import use_plugin, imread, imsave
import matplotlib.pyplot as plt

from coords2d import Coords2D
from reconstructor import Reconstruction, load_segmentation_maps

def shades_of_jop():
    c1 = random.randint(127, 255) 
    c2 = random.randint(0, 127) 
    c3 = random.randint(0, 255) 

    return tuple(random.sample([c1, c2, c3], 3))

SHADES_OF_JOP_USED = set()
def shades_of_jop_unique():
    jop = shades_of_jop()
    if jop in SHADES_OF_JOP_USED:
        jop = shades_of_jop_unique()
    else:
        SHADES_OF_JOP_USED.add(jop)
    return jop

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def generate_cell_collections(images):

    return [cell_dict_from_image_array(im) for im in images]

def load_intensity_data(intensity_dir):
    
    image_files = os.listdir(intensity_dir)

    image_files = sorted_nicely(image_files)

    idata = [imread(os.path.join(intensity_dir, im_file))
             for im_file in image_files]

    return idata

def plot_reconstruction(seg_dir, start_z):
    use_plugin('freeimage')

    smaps = load_segmentation_maps(seg_dir)

    xdim, ydim = smaps[0].im_array.shape

    r = Reconstruction(smaps, start=start_z) 

    end_z = 15
    for z in range(start_z, end_z-1):
        r.extend(z)

    rcells = r.cells_larger_then(3)

    das = {z: np.zeros((xdim, ydim, 3), dtype=np.uint8)
           for z in range(start_z, end_z)}

    for rcell in rcells:
        c = shades_of_jop_unique()
        for z, cellslice in rcell.slice_dict.items():
            das[z][cellslice.coord_list] = c

    for z, ia in das.items():
        imsave("da%d.tif" % z, ia)
        

def main():
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seg_dir', help="Path to directory containing segmented images")

    args = parser.parse_args()

    recons = plot_reconstruction(args.seg_dir, 0)

    

if __name__ == "__main__":
    main()
