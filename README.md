# POETSS
*Photometric Optimal Extraction of Time Series Spectra*

This is a collection of routines that optimally extracts photometry
from spectral time series from space telescopes where the PSF and pointing
are stable.

These routines assume a basic reduction of data has been made and that the data 
are in the following format:

A) Flux data in a numpy array cube, with
(frame#, row# (spatial direction), column# (wavelentgth direction))

B) The corresponding noise cube, containing 1 std uncertainties for all data

C) A bad pixelmap, i.e. a 2D map of pixels that are bad in all frames. The format is
a boolean fram where pixel is True if bad, False otherwise.

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
