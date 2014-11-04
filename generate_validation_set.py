"""Generate images for validating the segmentation."""

import os
import argparse
import random

import numpy as np
from skimage.io import use_plugin, imread, imsave
from skimage.morphology import disk, square, binary_erosion

from libtiff import TIFF

from time import time

from reconstructor import (
    Reconstruction,
    load_segmentation_maps,
    sorted_nicely,
)
from reconstruct_and_measure import load_intensity_data

class ValidationSet(object):
    """Class for generating a validation set."""

    Z_FIRST = 3
    Z_LAST = 11

    def __init__(self, seg_dir, cell_wall_dir, measurement_dir, out_dir, out_prefix):
        use_plugin('freeimage')

        self.reconstructed_cells = []
        self.selected_points = []

        self.segmentation_maps = load_segmentation_maps(seg_dir)
        self.cell_wall_images = self.get_images(cell_wall_dir)
        self.measurement_images = self.get_images(measurement_dir)
        self.out_dir = out_dir
        self.out_prefix = out_prefix

        self.segment_me_dir = os.path.join(out_dir, 'segment_me')
        self.answer_dir = os.path.join(out_dir, 'answers')

        for d in (self.out_dir, self.segment_me_dir, self.answer_dir):
            if not os.path.isdir(d):
                os.mkdir(d)

        self.xdim, self.ydim = self.measurement_images[0].shape

        start = time()
        self.reconstruction = Reconstruction(self.segmentation_maps, start=0) 
        for z in range(0, len(self.segmentation_maps)-1):
            self.reconstruction.extend(z)
        elapsed = ( time() - start ) / 60
        print('Reconstruction done {} minutes.'.format(elapsed))

    def get_filename(self):
        """Return image file name."""
        rcell_id = self.reconstructed_cells[0].ID
        return '{}_rcell{}.tif'.format(self.out_prefix, rcell_id)

    def get_images(self, directory):
        """Return list of images sorted nicely."""
        image_files = os.listdir(directory)
        image_files = sorted_nicely(image_files)
        idata = [imread(os.path.join(directory, im_file))
                 for im_file in image_files]
        return idata

    def get_random_plane(self):
        """Return a random plane (z) integer."""
        return random.randint(ValidationSet.Z_FIRST, ValidationSet.Z_LAST)

    def get_random_point(self):
        """Return a random point (x, y) in a plane."""
        x = random.randint(0, self.xdim-1)
        y = random.randint(0, self.ydim-1)
        return x, y

    def pick_reconstructed_cells(self):
        """Pick cells to reconstruct."""
        while len(self.reconstructed_cells) < 1:
            z = self.get_random_plane()
            x, y = self.get_random_point()
            cell_id = self.segmentation_maps[z].cell_id_at((x,y))
            rcell = self.reconstruction.find_cell(z, cell_id)
            if (rcell is not None
                and rcell.z_extent > 2
                and rcell.z_extent < 5
                and rcell not in self.reconstructed_cells):
                print('rCell: {}'.format(rcell))
                self.selected_points.append( (x, y, z) )
                self.reconstructed_cells.append( rcell )

    def get_selected_points_image(self, z_plane):
        """Return selected points image for a particular z-stack."""
        size = 6
        not_in_plane_symbol = square(size)
        in_plane_symbol = disk(size)

        def get_x_y_pixel_values(x, y, salem):
            for i, ys in enumerate(salem):
                for j, vals in enumerate(ys):
                    pixel = salem[i,j] * 255
                    xval = x + i - size
                    yval = y + j - size
                    yield (xval, yval, pixel)

        im = np.zeros((self.xdim, self.ydim), dtype=np.uint8)
        for x, y, z in self.selected_points:
            if z == z_plane:
                data = get_x_y_pixel_values(x, y, in_plane_symbol)
            else:
                data = get_x_y_pixel_values(x, y, not_in_plane_symbol)
            for x, y, pixel in data:
                im[x, y] = pixel
        return im

    def generate_augmented_image(self):
        """Generate augmented image with x,y,z points."""
        out_fname = os.path.join(self.segment_me_dir, self.get_filename())
        tif = TIFF.open(out_fname, 'w')
        for z, (cell_wall_im, measurement_im) in enumerate(zip(self.cell_wall_images,
                                                self.measurement_images)):
            point_im = self.get_selected_points_image(z)
            aug_im_rgb = np.array([cell_wall_im, measurement_im, point_im])
            tif.write_image(aug_im_rgb, write_rgb=True)

    def get_segmentation_outline_image(self, z_plane):
        """Return segmentation_outline_image for a particular z-stack."""

        def get_outline(im):
            """Return segmentation outline."""
            binary_mask_im = im != 0
            salem = disk(4)
            erosion_mask_im = binary_erosion(binary_mask_im, salem)
            outline_mask_im = np.logical_xor(binary_mask_im, erosion_mask_im)
            return im * outline_mask_im

        im = np.zeros((self.xdim, self.ydim), dtype=np.uint8)
        for rcell in self.reconstructed_cells:
            try:
                cell_slice = rcell.slice_dict[z_plane]
            except KeyError:
                continue
            for x, y in zip(cell_slice.x_coords, cell_slice.y_coords):
                im[x, y] = 255
        return get_outline(im)


        return im

    def generate_answer_image(self):
        """Generate image augmented with the answer."""
        out_fname = os.path.join(self.answer_dir, self.get_filename())
        tif = TIFF.open(out_fname, 'w')
        for z, (cell_wall_im, measurement_im) in enumerate(zip(self.cell_wall_images,
                                                self.measurement_images)):
            seg_im = self.get_segmentation_outline_image(z)
            aug_im_rgb = np.array([cell_wall_im, measurement_im, seg_im])
            tif.write_image(aug_im_rgb, write_rgb=True)

    def generate_validation_set(self):
        """Generate the validation images."""
        self.pick_reconstructed_cells()
        self.generate_augmented_image()
        self.generate_answer_image()

def select_random_root():
    """Return tuple of input file paths to a random root."""
    base_dir = '/localscratch/olssont/flc_venus'

    def get_unpacked_base(e, t, s):
        return os.path.join(base_dir, 'unpacked_data', 'Expt47', e, t, s) 
    def get_cell_wall_dir(e, t, s):
        d = get_unpacked_base(e, t, s)
        return os.path.join(d, 'cellwall')
    def get_venus_dir(e, t, s):
        d = get_unpacked_base(e, t, s)
        return os.path.join(d, 'venus')
    def get_seg_dir(e, t, s):
        exp = '{}_results'.format(e)
        return os.path.join(base_dir, exp, t, s, 'Master', 'Segmentation')
    def get_out_prefix(e, t, s):
        return '{}_{}_{}'.format(e, t, s)

    e = random.choice(['SDB265', 'SDB281'])
    t = random.choice(['2WT7', '4WT7', '6WT7', '8WT7', '10WT7'])
    s = random.choice(['S0', 'S1', 'S2', 'S3', 'S4', 'S5'])

    return (
        get_seg_dir(e, t, s),
        get_cell_wall_dir(e, t, s),
        get_venus_dir(e, t, s),
        get_out_prefix(e, t, s)
    )
        

def get_validation_set_size(out_dir):
    d = os.path.join(out_dir, 'answers')
    try:
        fnames = os.listdir(d)
    except OSError:
        return 0
    return len(fnames)
    
def generate_validation_set(out_dir, size):
    print('type(size)', type(size))
    print('size', size)
    print('out_dir', out_dir)
    print(get_validation_set_size(out_dir))
    print(get_validation_set_size(out_dir) < size)
    while get_validation_set_size(out_dir) < size:
        s_dir, cw_dir, m_dir, out_prefix = select_random_root()
        print(s_dir, cw_dir, m_dir, out_prefix)
        try:
            vs = ValidationSet(s_dir, cw_dir, m_dir, out_dir, out_prefix)
            vs.generate_validation_set()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_dir', help="Path to output directory")
    parser.add_argument('size', type=int, help="Size of the validation set")
    args = parser.parse_args()
    generate_validation_set(args.out_dir, args.size)


if __name__ == '__main__':
    main()
