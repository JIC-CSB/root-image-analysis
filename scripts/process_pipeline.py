import os.path
import argparse

from workflow import (
    run,
    ManyToManyNode,
    ManyToOneNode,
    BaseSettings,
    setup_logger,
)
from object_mask import generate_object_mask
from apply_mask import apply_mask
from segmentation import full_segment_image
from remove_border_segmentations import remove_border_segmentations
from reconstruct_and_measure import reconstruct_and_measure
from sum_segmentation_dir import sum_segmentation_dir
from segmentation_outline import generate_segmentation_outline
from generate_heatmap import generate_heatmap

import logging
from time import time

node_logger = logging.getLogger('workflow')
node_logger.setLevel(logging.INFO)

script_logger = setup_logger(__name__)
script_logger.setLevel(logging.INFO)


HERE = os.path.dirname(os.path.realpath(__file__))

def log_msg(self, in_name):
    """Print a log message."""
    msg = '{} about to process input {}.'.format(self.__class__.__name__,
                                                 in_name)
    script_logger.info(msg)

class RootMask(ManyToManyNode):
    """Generates a mask of the root from the cell wall image."""
    class Settings(BaseSettings):
        min_size = 1000
        dilate = 6
    
    def execute(self, task_input):
        generate_object_mask(task_input.input_file, task_input.output_file,
                             task_input.settings.min_size,
                             task_input.settings.dilate)

class ApplyMask(ManyToManyNode):
    """Apply the root mask to the cell wall image."""
    def process(self):
        for cell_wall_fname, mask_fname in self.input_files:
            out_fname = self.get_output_file(cell_wall_fname)
            log_msg(self, (cell_wall_fname, mask_fname))
            if out_fname.exists \
            and out_fname.is_more_recent_than(cell_wall_fname) \
            and out_fname.is_more_recent_than(mask_fname):
                script_logger.info('Output file {} exists; skipping.'.format(out_fname))
                continue
            script_logger.info('Processing input.')
            apply_mask(cell_wall_fname, mask_fname, out_fname)
            script_logger.info('Done! Ouput file: {}.'.format(out_fname))

class Segmentation(ManyToManyNode):
    """Segment the masked cell wall image."""
    class Settings(BaseSettings):
        fiji_exe = '/usr/users/a5/olssont/software/source/fiji/Fiji.app/ImageJ-linux64'  
        fiji_script = os.path.join(HERE, 'watershed.ijm')
        min_num_pixels = 200

    def execute(self, task_input):
        full_segment_image(task_input.input_file,
                           task_input.output_file,
                           task_input.settings.fiji_exe,
                           task_input.settings.fiji_script,
                           task_input.settings.min_num_pixels)
        

class RemoveBorderSegmentations(ManyToManyNode):
    """Remove segments that touch the image border."""
    def execute(self, task_input):
        remove_border_segmentations(task_input.input_file, task_input.output_file)

class NewMeasurement(ManyToOneNode):
    """Measure the mean, quartile and best intensities of the segmented cells."""
    class Settings(BaseSettings):
        start_z = None
        end_z = None
    def process(self):
        segmentation_dir = self.input_obj[0].output_directory
        venus_dir = self.input_obj[1]
        out_fname = self.output_file
        log_msg(self, (segmentation_dir, venus_dir))
        if out_fname.exists \
        and out_fname.is_more_recent_than(self.input_obj[0].output_files[0]):
            script_logger.info('Output file {} exists; skipping.'.format(out_fname))
            return
        script_logger.info('Processing input.')
        reconstruct_and_measure(segmentation_dir,
                                venus_dir,
                                self.output_directory,
                                out_fname,
                                self.settings.start_z,
                                self.settings.end_z)
        script_logger.info('Done! Ouput file: {}.'.format(out_fname))

class ReconstrucitonOutline(ManyToManyNode):
    """Generate segmentation outlines using the reconstructed cells."""
    def get_tasks(self):
        tasks = []
        input_dir = self.input_obj.output_directory
        for fname in os.listdir(input_dir):
            input_fn = os.path.join(input_dir, fname)
            output_fn = self.get_output_file(fname)
            tasks.append((input_fn, output_fn))
        return tasks

    def execute(self, task_input):
        generate_segmentation_outline(task_input[0], task_input[1])

class Heatmap(ManyToOneNode):
    """Generate segmentation outlines using the reconstructed cells."""
    class Settings(BaseSettings):
        start_z = None
        end_z = None
    def process(self):
        segmentation_dir = self.input_obj[0].output_directory
        venus_dir = self.input_obj[1]
        out_fname = self.output_file
        log_msg(self, (segmentation_dir, venus_dir))
        if out_fname.exists \
        and out_fname.is_more_recent_than(self.input_obj[0].output_files[0]):
            script_logger.info('Output file {} exists; skipping.'.format(out_fname))
            return
        script_logger.info('Processing input.')
        generate_heatmap(segmentation_dir,
                         venus_dir,
                         self.output_directory,
                         out_fname,
                         self.settings.start_z,
                         self.settings.end_z)
        script_logger.info('Done! Ouput file: {}.'.format(out_fname))


class Master(ManyToOneNode):
    """End to end workflow."""
    def configure(self):
        cell_wall_dir = self.input_obj[0]
        venus_dir = self.input_obj[1]
        results_csv_fn = self.output_obj

        root_mask_node = self.add_node(RootMask(cell_wall_dir))
        apply_mask_node = self.add_node(ApplyMask(
                                       input_obj=(cell_wall_dir, root_mask_node)))
        segmentation_node = self.add_node(Segmentation(apply_mask_node))
        remove_border_segmentation_node = self.add_node(RemoveBorderSegmentations(segmentation_node))
        heatmap_node = self.add_node(Heatmap(
                                       input_obj=(remove_border_segmentation_node, venus_dir),
                                       output_obj="dummy.png"))
        new_measurement_node = self.add_node(NewMeasurement(
                                       input_obj=(remove_border_segmentation_node, venus_dir),
                                       output_obj=results_csv_fn))
        reconstruction_outline = self.add_node(ReconstrucitonOutline(
                                               input_obj=new_measurement_node))

def process_pipeline(root_dir, out_dir, mapper):
    cell_wall_dir = os.path.join(root_dir, 'cellwall')
    venus_dir = os.path.join(root_dir, 'venus')
    output_file = os.path.join(out_dir, 'final_results.csv')

    master_node = Master(input_obj=(cell_wall_dir, venus_dir),
                         output_obj=output_file)
    master_node.output_directory = out_dir
    run(master_node, mapper)

def process_many_series(root_dir, out_dir, mapper):
    series_dirs = [d for d in os.listdir(root_dir) if d.startswith('S')]

    for sd in series_dirs:
        new_root_dir = os.path.join(root_dir, sd)
        new_out_dir = os.path.join(out_dir, sd)
        if not os.path.isdir(new_out_dir):
            os.mkdir(new_out_dir)
        script_logger.info('Processing series in: {}'.format(new_out_dir))
        process_pipeline(new_root_dir, new_out_dir, mapper)

def process_many_treatments(root_dir, out_dir, mapper):
    treatment_dirs = os.listdir(root_dir)

    for td in treatment_dirs:
        new_root_dir = os.path.join(root_dir, td)
        new_out_dir = os.path.join(out_dir, td)
        if not os.path.isdir(new_out_dir):
            os.mkdir(new_out_dir)
        script_logger.info('Processing treatment in: {}'.format(new_out_dir))
        process_many_series(new_root_dir, new_out_dir, mapper)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root_dir', help="Root of directory structure containing files to process")
    parser.add_argument('out_dir', help="Output directory")

    args = parser.parse_args()

    from multiprocessing import Pool
    num_workers = 5
    pool = Pool(num_workers)

    start = time()
#   process_many_treatments(args.root_dir, args.out_dir, map)
#   process_many_series(args.root_dir, args.out_dir, pool.map)
    process_pipeline(args.root_dir, args.out_dir, mapper=pool.map)

    elapsed = (time() - start) / 60
    script_logger.info('Time taken {:.3f} minutes, using {} cores.'.format(elapsed, num_workers))
    


if __name__ == "__main__":
    main()
