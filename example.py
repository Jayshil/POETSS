# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:55:17 2023

@author: Alexis Brandeker (alexis@astro.su.se)

Test POETSS on mock data, illustrates how to use code

First generate some mock data. The mock object produces
a data cube representing 2D frames of spectra in a time
series. The mock object contains all relevant parameters
of the simulation, including offsets and bad pixel map
etc. 

The use POETSS to identify bad pixels and outliers,
produce a clean version of the data, measure
the centre-of-flux spectral trace locations, fitting
a polynomial to the trace. Since POETSS works  
using per-pixel photometry rather than trying to 
find a PSF, the centring of the trace is not critical
-- but finding the jitter is important.

Then extract the region around the trace which will be 
used for photometry. For each pixel in the resultng slice, 
fit a polynomial as a relation between the measured flux
and the jitter dx

"""

if __name__ == '__main__':

    import poetss
    import mock
    from time import time

    # Global time variables used by timeit() below
    T0 = T1 = time()

    def timeit(label=''):
        """Helper function to give time stamps and
        to estimate execution times
        """
        global T0, T1
        T = time()
        print('{:s}: T = {:.3f} s, DT = {:.3f} s'.format(label, T-T0, T-T1))
        T1 = T

    # (Number of frames, rows (spatial), columns (wavelenth))
    shape = (1000, 64, 1200) 
    pd = mock.PoetssData(shape)
    timeit('Initiate mock object')

    data, noise = pd.generate(occ_depth=1000)
    timeit('Generate data')

    nandata = poetss.cr2nan(data, pd.bad_map)
    timeit('Mark bad pixels and strong outliers as NaN')

    clean = poetss.clean_nan(nandata)
    timeit('Clean NaN')

    cent_mat = poetss.find_trace_cof(clean)
    timeit('Trace measurement')

    trace, dx = poetss.fit_multi_trace(cent_mat)
    timeit('Trace fit')

    ext_rad = 5 # Extraction radius
    clean_cut = poetss.extract_trace(clean, trace, psf_rad=ext_rad)
    data_cut = poetss.extract_trace(nandata, trace, psf_rad=ext_rad)
    noise_cut = poetss.extract_trace(noise, trace, psf_rad=ext_rad)
    timeit('Extract trace')

    coeffs = poetss.define_poly_coeff(clean_cut, dx, deg=1)
    timeit('Fitting coeffs')

    lc, lc_err, synth = poetss.photometry(data_cut, noise_cut, coeffs, dx)
    timeit('Spectro-photometric time series')

    sp = poetss.spectrum(coeffs)
    timeit('Average spectrum')
