"""Run a segmentation using a fiji script."""

import os
import os.path
import argparse
import tempfile

from PIL import Image
import numpy as np
from scipy.ndimage import measurements

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def segment_image(input_file, fiji_exe, fiji_script):
    """Return a segmented numpy image.
    
    Runs fiji in headless mode."""

    with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp_fh:
        output_file = tmp_fh.name
    cmd = '{} --headless -macro {} {}:{}'.format( fiji_exe, fiji_script,
                                                  input_file, output_file)

    print(cmd)
    os.system(cmd)
    print('Back in Python...')

    numpy_im = np.array(Image.open(output_file).convert('L'))
    os.unlink(output_file)
    return numpy_im 

def color_objects(numpy_im, min_num_pixels):
    """Return a segmented PIL image with each object colored differently."""
    numpy_im = 1*(numpy_im<128)  # make sure the image is binary
    labels, num_objects = measurements.label(numpy_im)
    print 'Number of objects: {}'.format(num_objects)

    # Filter out objects that do not have enough pixels.
    num_objects_passing_size_filter = 0
    for i in range(num_objects):
        bool_mask = labels==i+1
        size = measurements.sum( labels==i+1 )
        if size < min_num_pixels:
            # Set unwanted pixels to 0
            labels = labels * np.uint8( labels!=i+1 )
        else:
            num_objects_passing_size_filter += 1
    print 'Number of objects after size filter: {}'.format(
                                num_objects_passing_size_filter)
            
    pil_im = Image.fromarray(np.uint8(labels))
    return pil_im

def full_segment_image(input_file, output_file, fiji_exe, fiji_script, min_num_pixels):
    numpy_im = segment_image(input_file, fiji_exe, fiji_script)
    pil_im = color_objects(numpy_im, min_num_pixels)
    pil_im.save(output_file)

def main(args):
    "Main logic of the script."
    numpy_im = segment_image(args.input_file, args.fiji_exe, args.fiji_script)
    pil_im = color_objects(numpy_im, min_num_pixels=args.min_num_pixels)
    pil_im.save(args.output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='Input image file')
    parser.add_argument('output_file', help='Output image file')
    parser.add_argument('-m', '--min_num_pixels', default=200, type=int,
                        help='Minimum particle size')
    parser.add_argument('-s', '--fiji_script', default=None, help='Location of fiji script')
    parser.add_argument('-f', '--fiji_exe', default='fiji', help='Location of fiji executable')

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        parser.error('No such file: {}'.format(args.input_file))

    if args.fiji_script is None:
        args.fiji_script = os.path.join(SCRIPT_DIR, 'watershed.ijm')
    if not os.path.isfile(args.fiji_script):
        parser.error('No such file: {}'.format(args.fiji_script))

    main(args)
