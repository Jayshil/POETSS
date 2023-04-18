# POETSS
*Photometric Optimal Extraction of Time Series Spectra*
<i>Author: Alexis Brandeker ([alexis@astro.su.se](mailto:alexis@astro.su.se))</i>

This is a collection of routines that optimally extracts photometry
from spectral time series from space telescopes where the PSF and pointing
are stable.

These routines assume a basic reduction of data has been made and that the data 
are in the following format:

A) Flux data in a numpy array cube, with
(frame#, row# (spatial direction), column# (wavelentgth direction))

B) The corresponding noise cube, containing uncertainties (1 std) for all data

C) A bad pixelmap, i.e. a 2D array of pixels that are bad in all frames.
The format is a boolean 2D numpy array (row#, column#) fram where pixel
is True if bad, False otherwise.

The routines are then used to 

1) Identify outliers (cosmic rays)
2) Determine the trace of the spectrum in all frames, finding the relative
    offset in the spatial direction (due to jitter)
3) Shift the region around the trace into a matrix where the spectral trace
    is approximately parallel to the rows
4) Define linear correlation between pixel values and dx
5) Extract photometry and error per column in frame

Also included is a class to mock data. Example code on how to run POETSS
is given by the end of poetss.py.

## Installation
Installation for `POETSS` can be done using `setup.py` file in the repository, by following commands below:

```
git clone https://github.com/alphapsa/POETSS.git
cd pycdata
python setup.py install
```

There you are! You are now ready to use this package!