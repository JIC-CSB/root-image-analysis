"""Generate partial 3D reconstruction from series of 2D slices"""


import re
import os
import sys
import argparse
from pprint import pprint

import numpy as np
from skimage.io import use_plugin, imread, imsave
import matplotlib.pyplot as plt

from coords2d import Coords2D
from reconstructor import Reconstruction, load_segmentation_maps

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

def heatmap_stuff():
    da = np.zeros((xdim, ydim, 3), dtype=np.uint8)

    ilist = []
    imax = 4
    imin = 0.9
    for rcell in rcells:
        if 7 in rcell.slice_dict:
            intensity = rcell.measure_mean_intensity(idata)
            ilist.append(intensity)
            gval = 255 * (intensity - imin) / (imax - imin)
            gval = min(255, gval)
            da[rcell.slice_dict[7].coord_list] = [255-gval, gval, 0]

    print max(ilist), min(ilist)


    imsave("myr.tif", da)

def reconstruct_and_measure(seg_dir, measure_dir, results_file, start_z=0):
    use_plugin('freeimage')

    smaps = load_segmentation_maps(seg_dir)
    idata = load_intensity_data(measure_dir)

    xdim, ydim = idata[0].shape

    r = Reconstruction(smaps, start=start_z) 

    for z in range(start_z, len(smaps)-1):
    #for z in range(start_z, 12):
        r.extend(z)

    # r = Reconstruction(smaps, start=6) 
    # for z in range(6, 9):
    #     r.extend(z)

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
    parser.add_argument('results_file', help="Filename to which results should be output")

    args = parser.parse_args()

    recons = reconstruct_and_measure(args.seg_dir, args.measure_dir, args.results_file, 0)

    

if __name__ == "__main__":
    main()
