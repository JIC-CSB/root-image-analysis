# Root reconstruction and fluorescence measurement

This code implements a quantitative image analysis pipeline for the reconstruction of root cell volumes from confocal microscopy data. These reconstructed cell volumes can be used to measure fluorescence intensity in other channels in the confocal images.

## Using the code

If you're interested in using the code, we'd strongly suggest that you get in contact with us (we'd love to help!).

## Requirements

The following python packages are needed to run the code:

* numpy
* scipy
* matplotlib
* PIL
* openCV (with python binding) 
* scikit-image < 0.11
* libtiff
* freeimage

and you'll also need Fiji (the ImageJ implementation).

## Installation notes

```
virtualenv env
source env/bin/activate
pip install numpy
pip install scipy
pip install "scikit-image<0.11"
pip install libtiff
cp /lib64/python2.7/site-packages/cv2.so env/lib64/python2.7/
```

Set path to ``fiji`` in ``root-image-analysis/scripts/process_pipeline.py``.
