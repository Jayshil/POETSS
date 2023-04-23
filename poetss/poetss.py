# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 10:42:52 2023

@author: Alexis Brandeker (alexis@astro.su.se)

Photometric Optimal Extraction of Time Series Spectra (POETSS)

This is a collection of routines that optimally extracts photometry
from spectral time series from space telescopes where the PSF and pointing
are stable.

These routines assume a basic reduction have been made and that the data 
are on the following format:

A) flux data in a numpy array cube with
(frame#, row# (spatial direction), column# (wavelentgth direction))

B) the corresponding noise cube, containing uncertainties (1 std) for all data

C) a bad pixelmap, i.e. a 2D array of pixels that are bad in all frames.
The format is a boolean 2D numpy array (row#, column#) fram where pixel
is True if bad, False otherwise.

The routines are then used to 

1) Identify outliers (cosmic rays))
2) Determine the trace of the spectrum in all frames, finding the relative
    offset in the spatial direction (due to jitter)
3) Shift the region around the trace into a matrix where the spectral trace
    is approximately parallel to the rows
4) Define linear correlation between pixel flux values and jitter offset dx
5) Extract photometry and error per column in frame

"""
import numpy as np
import warnings


def cr2nan(data, bad_map, clip=5, niter=5):
    """ 
    Replaces outliers with NaN assuming there are only
    statistical differenes between frames, and defines
    anything more than clip std away per pixel as an outlier
    
    data is 3D array in format [frame#, row#, col#]
    bad_map is 2D array in format [row#, col#]] where
           True is bad pixel and False is good pixel
    clip is how much a cvalue should deviate to be defined as an outlier
    niter is the maxium number of iterations in sigma-clipping algorithm
    """
    indata = data.copy()
    indata[:, bad_map] = np.nan
    indata[0, bad_map] = 0      # To avoid pixels being NaN in all frames
    nandata = indata.copy()
    ind0 = np.zeros(indata.shape, dtype='?')
    
    for n in range(niter):
        s = np.nanstd(nandata, axis=0)
        m = np.nanmedian(nandata, axis=0)
        nandata = indata.copy()
        ind1 = np.abs(nandata-m) > clip*s
        print('Iter {:d}/{:d} masked: {:d} = {:.2f}%'.format(n+1, niter,
                    np.sum(ind1), 100.0*np.sum(ind1)/np.prod(data.shape)))
        nandata[ind1] = np.nan
        if n > 2 and np.prod(ind0 == ind1):
            break
        ind0 = ind1
    nandata[0,bad_map] = np.nan

    return nandata
 

def clean_nan(nandata, max_iter=50, N_chunks=8):
    """Replace NaN data points in a cube by interpolation. In a first step
    avoid interpolating across rows (since flux gradients are 
    generally greater there), but if any nan are left (can happen in case
    of bad rows), interpolate along all axes for a maxiumum of max_iter
    iterations.
    To ease memory requirements, the data can be divided up in N_chunks
    number of chunks.
    """
    clean = np.zeros_like(nandata)
    chunk = np.array(np.linspace(0,len(clean), N_chunks+1), dtype=int)
    
    for n in range(len(chunk)-1):
        clean[chunk[n]:chunk[(n+1)]] = _replace_nan(nandata[chunk[n]:chunk[(n+1)]],
                                                   max_iter=2, axis=(0,2))
    if np.sum(np.isnan(clean)) == 0:
        return clean
    for n in range(len(chunk)-1):
        clean[chunk[n]:chunk[(n+1)]] = _replace_nan(clean[chunk[n]:chunk[(n+1)]],
                                                   max_iter=max_iter)
    return clean


def find_trace_cof(clean_cube, margin=5):
    """ Use centre-of-flux to measure the trace for each frame and column.
    margin is how many pixels outside of the trace should be considered.
    """
    cube = clean_cube.copy()
    cube[cube<0] = 0 
    cube /= np.maximum(1, np.max(cube, axis=1))[:,None,:]

    border = int(cube.shape[2]/10)
    m = np.median(cube, axis=0)
    start_m = np.mean(m[:,:border], axis=1)
    start_x = np.where(start_m == np.max(start_m))[0][0]
    end_m = np.mean(m[:,-border:], axis=1)
    end_x = np.where(end_m == np.max(end_m))[0][0]
    dx = max(np.abs(end_x-start_x), margin)
    x0 = max(int(min(start_x, end_x)-dx), 0)
    x1 = min(int(max(start_x, end_x)+dx), cube.shape[1])
    
    # Compute centre of flux for each column
    row = np.arange(cube.shape[0])
    cent_mat = (np.sum(cube[:, x0:x1, :]*row[None, x0:x1, None], axis=1) / 
              np.maximum(np.sum(cube[:,x0:x1,:], axis=1), 1e-5))
    return cent_mat


def fit_multi_trace(cent_mat, deg=2, clip=3):
    """Fits a combined trace to the centers 
    measured in cent_mat (#frame, #column)
    deg is degree used in fitting polynomial,
    clip is significance used for clipping
    Returns trace (array with position for each column)
    and dx, array with offsets per frame.
    """
    med_cent = np.median(cent_mat, axis=0)    
    dx = np.median(cent_mat-med_cent[None,:], axis=1)
    cent_cent = cent_mat - dx[:,None]
    coeffs = np.zeros((len(cent_cent), deg+1))
    
    for n in range(len(cent_cent)):
        coeffs[n] = _fit_trace_poly(cent_cent[n], deg=deg, clip=clip)
        
    c = np.mean(coeffs, axis=0)
    p = np.poly1d(c)
    w = np.arange(cent_mat.shape[1])
    trace = p(w)
    dx = np.median(cent_mat-trace[None,:], axis=1)

    return trace, dx


def extract_trace(data_cube, trace, psf_rad=5):
    """Extracts a slit of radius psf_rad around the
    trace in the data_cube, and returns a new cube
    of same #frames and #columns, but with the slit length
    # of rows.
    """
    slitlen = int(2*psf_rad+1)
    x0 = np.array(trace-psf_rad, dtype=int)
    x1 = x0 + slitlen
    out_cube = np.zeros((data_cube.shape[0], slitlen, len(trace)))
    for n in range(len(trace)):
        out_cube[:, :, n] = data_cube[:, x0[n]:x1[n], n]
    return out_cube


def define_poly_coeff(clean_cut, dx, deg=1):
    """For each pixel in the frame, this defines polynomial coefficients
    (flux as a function of dx). Returns array of coefficients of 
    format (#coeffs per poly, #rows, #cols)
    """
    coeffs = np.zeros((deg+1, clean_cut.shape[1], clean_cut.shape[2]))
    
    for n in range(clean_cut.shape[2]):    
        coeffs[:, :, n] = np.polyfit(dx, clean_cut[:,:,n], deg=deg)
    return coeffs


def photometry(data, noise, poly_coeffs, dx):
    """Performs photometry on data cube of format (frame#, row#, col#),
    where chaning row changes spatial direction and changing column changes
    wavelength.
    The poly_coeffs are polynomial coefficients for each pixel in frame,
    determining how the sensitivity changes with offset dx
    dx is an array of 1D offset along spatial direction for all frames in
    cube
    Returns 
    """
    noise[np.isnan(data)] = np.nan
    base = np.ones_like(data)
    for n in range(data.shape[1]):
        for m in range(data.shape[2]):
            p = np.poly1d(poly_coeffs[:,n,m])
            base[:, n, m] = p(dx)
    
    weighted_data = data*base/noise**2
    weights = (base/noise)**2
    weight_sum = np.nansum(weights, axis=1)

    lightcurve = np.nansum(weighted_data, axis=1)/weight_sum
    lightcurve_err = 1/weight_sum**.5    
    synthspec = base*lightcurve[:,None,:]
    
    return lightcurve, lightcurve_err, synthspec


def spectrum(coeffs):
    """Returns the stellar spectrum as defined by the polynomial
    coefficients.
    """
    shape = coeffs.shape[1:]
    spec2D = np.ones(shape)
    for n in range(shape[0]):
        for m in range(shape[1]):
            p = np.poly1d(coeffs[:,n,m])
            spec2D[n, m] = p(0)
    return np.sum(spec2D, axis=0)


def white_light(lightcurve, lightcurve_err):
    """Computes the white light time series from a spectral time series,
    using weighted average. Returns the white light curve and its error
    """
    ssum = np.sum(1/lightcurve_err**2, axis=1)
    whl = np.sum(lightcurve/lightcurve_err**2, axis=1) / ssum
    whl_err = 1/ssum**.5
    return whl, whl_err


# Helper function

def _fit_trace_poly(cent_vec, deg=2, clip=3, niter=10):
    """Fits a polynomial to the measured positions. Uses
    sigma-clipping to remove outliers from fit
    """
    w = np.arange(len(cent_vec))
    c = np.polyfit(w, cent_vec, deg=deg)
    p = np.poly1d(c)
    sel0 = np.ones(len(cent_vec), dtype='?')
    for n in range(niter):
        s = np.std(cent_vec[sel0]-p(w[sel0]))
        sel1 = np.abs(cent_vec-p(w)) <= clip*s
        c = np.polyfit(w[sel1], cent_vec[sel1], deg=deg)
        if np.prod(sel0==sel1) == 1:
            break
        p = np.poly1d(c)
        sel0 = sel1
    return c


def _replace_nan(data, max_iter=50, axis=None):
    """Replaces NaN-entries by mean of neighbours.
    Iterates until all NaN-entries are replaced or
    max_iter is reached. Works on N-dimensional arrays.
    If axis tuple is defined, interpolations are made only
    along those axes.
    Memory intensive for large cubes.
    """
    if axis is None:
        axis = tuple(range(data.ndim))
    nan_data = data.copy()
    shape = np.append([2*len(axis)], data.shape)
    interp_cube = np.zeros(shape)
    shift0 = np.zeros(len(axis), int)
    shift0[0] = 1
    shift = []
    for n in range(len(axis)):
        shift.append(tuple(np.roll(-shift0, n)))
        shift.append(tuple(np.roll(shift0, n)))
    for _j in range(max_iter):
        for n in range(2*len(axis)):
            interp_cube[n] = np.roll(nan_data, shift[n], axis=axis)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_data = np.nanmean(interp_cube, axis=0)
        nan_data[np.isnan(nan_data)] = mean_data[np.isnan(nan_data)]
        if np.sum(np.isnan(nan_data)) == 0:
            break
    return nan_data
