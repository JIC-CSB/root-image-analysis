import os.path
import argparse

from workflow.node import MetaNode, ManyToManyNode, ManyToOneNode, BaseSettings
from object_mask import generate_object_mask
from apply_mask import apply_mask
from segmentation import full_segment_image
from generate_reconstruction import generate_reconstruction

import logging

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

HERE = os.path.dirname(os.path.realpath(__file__))

def log_msg(self, in_name):
    """Print a log message."""
    msg = '{} about to process input {}.'.format(self.__class__.__name__,
                                                 in_name)
    logger.info(msg)

class RootMask(ManyToManyNode):
    """Generates a mask of the root from the cell wall image."""
    class Settings(BaseSettings):
        min_size = 1000
        dilate = 6
    def process(self):
        for i, in_fname in enumerate(self.input_files):
            out_fname = self.get_output_file(in_fname)
            log_msg(self, in_fname)
            if not os.path.isfile(out_fname):
                logger.info('Processing input.')
                generate_object_mask(in_fname,
                                     out_fname,
                                     self.settings.min_size,
                                     self.settings.dilate)
                logger.info('Done! Ouput file: {}.'.format(out_fname))
            else:
                logger.info('Output file {} exists; skipping.'.format(out_fname))

class ApplyMask(ManyToManyNode):
    """Apply the root mask to the cell wall image."""
    def process(self):
        for cell_wall_fname, mask_fname in self.input_files:
            out_fname = self.get_output_file(cell_wall_fname)
            log_msg(self, (cell_wall_fname, mask_fname))
            if not os.path.isfile(out_fname):
                logger.info('Processing input.')
                apply_mask(cell_wall_fname, mask_fname, out_fname)
                logger.info('Done! Ouput file: {}.'.format(out_fname))
            else:
                logger.info('Output file {} exists; skipping.'.format(out_fname))

class Segmentation(ManyToManyNode):
    """Segment the masked cell wall image."""
    class Settings(BaseSettings):
        fiji_exe = '/usr/users/a5/olssont/software/fiji/Fiji.app/ImageJ-linux64'  
        fiji_script = os.path.join(HERE, 'watershed.ijm')
        min_num_pixels = 200

    def process(self):
        for in_fname in self.input_files:
            out_fname = self.get_output_file(in_fname)
            log_msg(self, in_fname)
            if not os.path.isfile(out_fname):
                logger.info('Processing input.')
                full_segment_image(in_fname, out_fname,
                                   self.settings.fiji_exe,
                                   self.settings.fiji_script,
                                   self.settings.min_num_pixels)
                logger.info('Done! Ouput file: {}.'.format(out_fname))
            else:
                logger.info('Output file {} exists; skipping.'.format(out_fname))

class Measurement(ManyToOneNode):
    """Measure the mean intensities of the segmented cells."""
    def process(self):
        segmentation_dir = self.input_obj[0].output_directory
        venus_dir = self.input_obj[1]
        out_fname = self.output_file
        log_msg(self, (segmentation_dir, venus_dir))
        if not os.path.isfile(out_fname):
            logger.info('Processing input.')
            generate_reconstruction(segmentation_dir, venus_dir, out_fname)
            logger.info('Done! Ouput file: {}.'.format(out_fname))
        else:
            logger.info('Output file {} exists; skipping.'.format(out_fname))
    
class Master(MetaNode):
    """End to end workflow."""

def process_pipeline(root_dir, out_dir):
    cell_wall_dir = os.path.join(root_dir, 'cellwall')
    venus_dir = os.path.join(root_dir, 'venus')
    results_fname = os.path.join(out_dir, 'results.csv')

    root_mask_node = RootMask(input_obj=cell_wall_dir)
    apply_mask_node = ApplyMask(input_obj=(cell_wall_dir, root_mask_node))
    segmentation_node = Segmentation(input_obj=apply_mask_node)
    measurement_node = Measurement(input_obj=(segmentation_node, venus_dir),
                                   output_obj=results_fname)

    master_node = Master()
    master_node.add_node(root_mask_node)
    master_node.add_node(apply_mask_node)
    master_node.add_node(segmentation_node)
    master_node.add_node(measurement_node)

    master_node.output_directory = out_dir
    master_node.run()

def process_many_series(root_dir, out_dir):
    series_dirs = [d for d in os.listdir(root_dir) if d.startswith('S')]

    for sd in series_dirs:
        new_root_dir = os.path.join(root_dir, sd)
        new_out_dir = os.path.join(out_dir, sd)
        if not os.path.isdir(new_out_dir):
            os.mkdir(new_out_dir)
        logger.info('Processing series in: {}'.format(new_out_dir))
        process_pipeline(new_root_dir, new_out_dir)

def process_many_treatments(root_dir, out_dir):
    treatment_dirs = os.listdir(root_dir)

    for td in treatment_dirs:
        new_root_dir = os.path.join(root_dir, td)
        new_out_dir = os.path.join(out_dir, td)
        if not os.path.isdir(new_out_dir):
            os.mkdir(new_out_dir)
        logger.info('Processing treatment in: {}'.format(new_out_dir))
        process_many_series(new_root_dir, new_out_dir)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root_dir', help="Root of directory structure containing files to process")
    parser.add_argument('out_dir', help="Output directory")

    args = parser.parse_args()

#   process_many_treatments(args.root_dir, args.out_dir)
#   process_many_series(args.root_dir, args.out_dir)
    process_pipeline(args.root_dir, args.out_dir)

if __name__ == "__main__":
    main()
