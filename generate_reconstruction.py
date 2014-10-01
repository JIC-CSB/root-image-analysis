"""Generate partial 3D reconstruction from series of 2D slices"""


import re
import os
import sys
import argparse

from pprint import pprint

import numpy as np

from skimage.io import use_plugin, imread

import matplotlib.pyplot as plt

from coords2d import Coords2D

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

class CellSlice(object):

    def __init__(self, ID, coord_list):
        self.ID = ID
        self.coord_list = coord_list
        self.pixel_area = len(coord_list[0])
        self.x_coords = coord_list[0]
        self.y_coords = coord_list[1]

    @property
    def centroid(self):
        return Coords2D(sum(self.x_coords), 
                        sum(self.y_coords)) / self.pixel_area

    @property
    def summary(self):
        return "<ID: %d, pixel_area: %d, centroid: %s>" % (self.ID, 
                                                           self.pixel_area, 
                                                           self.centroid)

    def __repr__(self):
        
        return "<CellSlice, ID %d>" % self.ID

class SegmentationMap(object):

    def __init__(self, image_file):
        self.im_array = imread(image_file)
        self.internal_cc = None

    @property
    def cells(self):
        if self.internal_cc is not None:
            return self.internal_cc
        else:
            self.internal_cc = cell_dict_from_image_array(self.im_array)
            return self.internal_cc

    def cell_at(self, position):
        x, y = position

        ID = self.im_array[x, y]
        if ID == 0:
            return None

        return self.cells[self.im_array[x, y]]

    @property
    def all_ids(self):
        with_zero = list(np.unique(self.im_array))
        with_zero.remove(0)
        return with_zero

    def coord_list(self, cID):
        return np.where(self.im_array == cID)

class Reconstruction(object):

    def __init__(self, smaps, start=0):
        self.smaps = smaps
        self.rcells = []
        self.lut = {}
        z = start
        for ID in smaps[z].all_ids:
            rcell = ReconstructedCell({z: self.smaps[z].cells[ID]})
            self.rcells.append(rcell)
            self.lut[(z, ID)] = rcell

    def extend(self, level):
        z = level
        matches = find_slice_links(self.smaps[z], self.smaps[z+1])

        for f, t in matches.iteritems():
            try:
                rcell = self.lut[(z, f)]
                rcell.add_slice(z+1, self.smaps[z+1].cells[t])
            except KeyError:
                rcell = ReconstructedCell({z+1: self.smaps[z+1].cells[t]})
                self.rcells.append(rcell)

            self.lut[(z+1, t)] = rcell

    def cells_larger_then(self, zext):
        return [rcell for rcell in self.rcells
                if rcell.z_extent >= zext]

    def save_to_file(self, filename):
        with open(filename, "w") as f:
            f.write('\n'.join([rcell.simple_string_rep()
                               for rcell in self.rcells]))

def cell_dict_from_image_array(i_array):

    cd = {cid: CellSlice(cid, np.where(i_array == cid))
          for cid in np.unique(i_array)}

    del(cd[0])

    return cd

def generate_cell_collections(images):

    return [cell_dict_from_image_array(im) for im in images]

def slice_from_same_cell(slice1, slice2):

    if slice1 is None or slice2 is None:
        return False

    area1 = slice1.pixel_area
    area2 = slice2.pixel_area

    dist = slice1.centroid.dist(slice2.centroid)

    area_ratio = float(area1) / area2

    if 0.5 < area_ratio < 1.5 and dist < 20:
        return True
    else:
        return False

def load_segmentation_maps(slice_dir):

    image_files = os.listdir(slice_dir)

    image_files = sorted_nicely(image_files)

    smaps = [SegmentationMap(os.path.join(slice_dir, im_file))
             for im_file in image_files]

    return smaps

def find_slice_links(smaps):

    matches = []
    for cell_slice in smaps[1].cells.values():
        md = {1: cell_slice}
        c = cell_slice.centroid
        candidate_slice = smaps[0].cell_at(c)
        if slice_from_same_cell(cell_slice, candidate_slice):
            md[0] = candidate_slice
            c_below = smaps[2].cell_at(c)
            if slice_from_same_cell(cell_slice, c_below):
                md[2] = c_below
                #matched.append(cell_slice)
        matches.append(md)

    return matches

def find_slice_links(slice_map1, slice_map2):
    matches = {}

    for cell_slice in slice_map1.cells.values():
        c = cell_slice.centroid
        candidate_slice = slice_map2.cell_at(c)
        if slice_from_same_cell(cell_slice, candidate_slice):
            #matches.append((cell_slice.ID, candidate_slice.ID))
            matches[cell_slice.ID] = candidate_slice.ID

    return matches

class ReconstructedCell(object):

    def __init__(self, slice_dict):
        self.slice_dict = slice_dict

    def add_slice(self, layer, cellslice):
        self.slice_dict[layer] = cellslice

    @property
    def pixel_area(self):
        return sum(cellslice.pixel_area
                   for cellslice in self.slice_dict.values())

    @property
    def z_extent(self):
        return len(self.slice_dict.keys())

    def measure_total_intensity(self, idata):
        total_intensity = 0

        for sID, cellslice in self.slice_dict.items():
            #z_correction = 1 + (0.03 * sID)
            z_correction = 1
            total_intensity += z_correction * sum(idata[sID][cellslice.coord_list])

        return total_intensity

    def measure_mean_intensity(self, idata):

        return float(self.measure_total_intensity(idata)) / self.pixel_area

    def __repr__(self):
        return "<ReconstructedCell: %s>" % self.slice_dict.__repr__()

    def simple_string_rep(self):
        return ",".join("%d:%d" % (sID, cellslice.ID)
                        for sID, cellslice
                        in self.slice_dict.items())

    @property
    def centroid(self):
        z = float(sum(self.slice_dict.keys())) / len(self.slice_dict)
        csum = sum([c.centroid for c in self.slice_dict.values()], 
                   Coords2D(0, 0))
        x, y = map(float, csum / len(self.slice_dict))
        return x, y, z 

def load_intensity_data(intensity_dir):
    
    image_files = os.listdir(intensity_dir)

    image_files = sorted_nicely(image_files)

    idata = [imread(os.path.join(intensity_dir, im_file))
             for im_file in image_files]

    return idata

def generate_reconstruction(seg_dir, measure_dir, results_file):
    use_plugin('freeimage')

    smaps = load_segmentation_maps(seg_dir)
    idata = load_intensity_data(measure_dir)

    r = Reconstruction(smaps, start=0) 

    for z in range(0, len(smaps)-1):
        r.extend(z)

    # r = Reconstruction(smaps, start=6) 
    # for z in range(6, 9):
    #     r.extend(z)

    rcells = r.cells_larger_then(3)

    with open(results_file, "w") as f:
        for rcell in rcells:
            x, y, z = rcell.centroid
            intensity = rcell.measure_mean_intensity(idata)
            volume = rcell.pixel_area
            zext = rcell.z_extent

            f.write("{},{},{},{},{},{}\n".format(
                intensity,
                x, y, z,
                volume,
                zext))

def main():
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seg_dir', help="Path to directory containing segmented images")
    parser.add_argument('measure_dir')

    args = parser.parse_args()

    #recons = generate_reconstruction(args.seg_dir, args.measure_dir, 'results')

    

if __name__ == "__main__":
    main()
