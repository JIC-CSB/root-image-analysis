import re
import os

import numpy as np
from skimage.io import use_plugin, imread

from coords2d import Coords2D
#import regionmap


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

class ReconstructedCell(object):
    """Pseudo 3D cell from contiguous z-stacks."""

    def __init__(self, slice_dict):
        self.slice_dict = slice_dict

    def add_slice(self, layer, cellslice):
        """Add ROI from another z-stack."""
        self.slice_dict[layer] = cellslice

    @property
    def pixel_area(self):
        """Return the total cell area across all the z-stacks."""
        return sum(cellslice.pixel_area
                   for cellslice in self.slice_dict.values())

    @property
    def z_extent(self):
        """Return the number of z-stacks in the reconstructed cell."""
        return len(self.slice_dict.keys())

    def measure_total_intensity(self, idata):
        """Return the total intensity across all the z-stacks."""
        total_intensity = 0
        for sID, cellslice in self.slice_dict.items():
            #z_correction = 1 + (0.03 * sID)
            z_correction = 1
            total_intensity += z_correction * sum(idata[sID][cellslice.coord_list])
        return total_intensity

    def measure_mean_intensity(self, idata):
        """Return the mean intensity of the reconstructed cell."""
        return float(self.measure_total_intensity(idata)) / self.pixel_area

    def measure_quartile_intensity(self, idata):
        """Return the quartile intensity of the reconstructed cell."""
        all_measurements = [idata[sID][cellslice.coord_list]
                            for sID, cellslice in self.slice_dict.items()]
        as_single_array = np.concatenate(all_measurements)
        n_frac = len(as_single_array) / 2
        return np.mean(as_single_array[n_frac:])

    def measure_best_slice(self, idata):
        """Return the intensity from the most intense slice."""
        slice_arrays = [idata[sID][cellslice.coord_list]
                        for sID, cellslice in self.slice_dict.items()]
        slice_means = [np.mean(sa) for sa in slice_arrays]
        best_slice = max(slice_means)
        best_z = np.array(slice_means).argmax()
        return best_slice, best_z
        
    def __repr__(self):
        return "<ReconstructedCell: %s>" % self.slice_dict.__repr__()

    def simple_string_rep(self):
        """Return a simple string representation of the reconstructed cell."""
        return ",".join("%d:%d" % (sID, cellslice.ID)
                        for sID, cellslice
                        in self.slice_dict.items())

    @property
    def centroid(self):
        """Return the centroid of the reconstructed cell."""
        z = float(sum(self.slice_dict.keys())) / len(self.slice_dict)
        csum = sum([c.centroid for c in self.slice_dict.values()], 
                   Coords2D(0, 0))
        x, y = map(float, csum / len(self.slice_dict))
        return x, y, z 


class CellSlice(object):
    """Slice of a cell in the x, y plane."""

    def __init__(self, ID, coord_list):
        self.ID = ID
        self.coord_list = coord_list
        self.pixel_area = len(coord_list[0])
        self.x_coords = coord_list[0]
        self.y_coords = coord_list[1]

    @property
    def centroid(self):
        """Return the centroid of the cell slice."""
        return Coords2D(sum(self.x_coords), 
                        sum(self.y_coords)) / self.pixel_area

    @property
    def summary(self):
        """Return summary string representation of the cell slice."""
        return "<ID: %d, pixel_area: %d, centroid: %s>" % (self.ID, 
                                                           self.pixel_area, 
                                                           self.centroid)

    def __repr__(self):
        return "<CellSlice, ID %d>" % self.ID

class SegmentationMap(object):
    """Container for the pseudo 3D reconstructed cells."""

    def __init__(self, image_file):
        use_plugin('freeimage')
        self.im_array = imread(image_file)
        # self.im_array = np.transpose(self.im_array)
        # cd = regionmap.load_image(image_file)
        # self.internal_cc = {cid: CellSlice(cid, clist)
        #                     for cid, clist in cd.items()}
        self.internal_cc = None

    @property
    def cells(self):
        """Return the dictionary of cell slices."""
        #return self.internal_cc
        if self.internal_cc is not None:
            return self.internal_cc
        else:
            self.internal_cc = cell_dict_from_image_array(self.im_array)
            return self.internal_cc

    def cell_at(self, position):
        """Return the cell at position (x, y)."""
        x, y = position

        ID = self.im_array[x, y]
        if ID == 0:
            return None

        return self.cells[self.im_array[x, y]]

    @property
    def all_ids(self):
        """Return all cell ids."""
        with_zero = list(np.unique(self.im_array))
        with_zero.remove(0)
        return with_zero

    def coord_list(self, cID):
        """Return list of coordinates for cell (ID)."""
        return np.where(self.im_array == cID)

class Reconstruction(object):
    """Pseudo 3D reconstruction."""

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
        """Add another z-stack to the reconstruction."""
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
        """Return cells which have more than zext z-stacks."""
        return [rcell for rcell in self.rcells
                if rcell.z_extent >= zext]

    def save_to_file(self, filename):
        """Save the reconstructed cells to file using simple cell representation."""
        with open(filename, "w") as f:
            f.write('\n'.join([rcell.simple_string_rep()
                               for rcell in self.rcells]))

def cell_dict_from_image_array(i_array):
    """Return dictionary of cell slices from an array of images."""
    cd = {cid: CellSlice(cid, np.where(i_array == cid))
          for cid in np.unique(i_array)}
    del(cd[0])
    return cd

def load_segmentation_maps(slice_dir):
    """Return list of segmentation maps from a directory of segmentations."""
    image_files = os.listdir(slice_dir)
    image_files = sorted_nicely(image_files)
    smaps = [SegmentationMap(os.path.join(slice_dir, im_file))
             for im_file in image_files]
    return smaps

def find_slice_links(slice_map1, slice_map2):
    """Return dictionary of matched cells between two slice maps."""
    matches = {}
    for cell_slice in slice_map1.cells.values():
        c = cell_slice.centroid
        candidate_slice = slice_map2.cell_at(c)
        if slice_from_same_cell(cell_slice, candidate_slice):
            matches[cell_slice.ID] = candidate_slice.ID
    return matches

def slice_from_same_cell(slice1, slice2):
    """Whether or not two cell slices are from the same cell."""
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
