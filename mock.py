# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 00:13:03 2023

@author: Alexis Brandeker (alexis@astro.su.se)

Photometric Optimal Extraction of Time Series Spectra (POETSS)

This is a data simulator class, used to generate data to be used
by POETSS

Effects included:
* Spectroscopic trace as polynomial
* Jitter of trace position per frame
* PSF changing size linearly with wavelength
* Bad pixels (for all frames)
* Cosmic rays (random location / strength)
* Spectrum with slope and randomly distributed spectral lines
  of random strength and width
* Convolution by spectral resolution
* Noise from photon statistics including source and sky background

NOT included:
* Dark current separate from sky background (e.g. hot pixels)
* Odd-even readout effects
* Detector non-linearity
* Uneven flat field (pixel response function)
* Pixel sensitivity evolution with time ("ramp")
* Other interfering sources in the field, scattered light ("ghosts")
* Detailed lightcurve 

"""
import numpy as np


class PoetssData():
    """Simulates data to be used for POETSS
    """
    
    def __init__(self, shape):
        """shape is shape to be used for data cube
        (#frames, #spatial pxiels, #wavelength pixels)

        """
        self.shape = shape
        self.start_fwhm = 2     # FWHM of PSF at start column, in pixels
        self.end_fwhm = 3       # FWHM of PSF at end column, in pixels
        self.jitter_std = 0.01  # Jitter std between frames, in pixels
        self.cr = 1e-3          # Cosmic Ray hit rate, per pixel
        self.bad = 5e-4         # Fraction of bad pixels
        self.bg_noise = 23      # Background noise, electrons per pixel
        self.slope = -3.1       # Slope of trace over detector, in pixels
        self.sed_fall = 0.5     # How much the SED falls to from the peak
        self.signal = 1e5       # Max number of photons per column


    def generate(self, occ_depth=100):
        """Occultation depth in ppm, assumed to occur during 
        central third of timeline
        Returns simulated data cube
        (frame number, spatial direction, wavelength direction)
        """
        x = np.arange(self.shape[1])
        w = np.arange(self.shape[2])
        
        self.star_spec = self._spectrum(w)
        self.planet_spec = self._spectrum(w) * occ_depth * 1e-6
        self.occ_start = int(self.shape[0]/3)
        self.occ_end = int(self.shape[0]*2/3)
        self.dx = self.jitter_std * np.random.randn(self.shape[0])        
        self.bad_map = np.random.rand(self.shape[1], self.shape[2]) <= self.bad

        X = x[None,:,None] - self._trace(w)[None,None,:] - self.dx[:,None,None]
        W = np.ones(self.shape)*w[None,None,:]
        signal = self._psf(W, X)
        signal[:self.occ_start] *= self.star_spec[None,None,:] + self.planet_spec[None,None,:]
        signal[self.occ_start:self.occ_end] *= self.star_spec[None,None,:]
        signal[self.occ_end:] *= self.star_spec[None,None,:] + self.planet_spec[None,None,:]
        signal *= self.signal

        variance = signal + self.bg_noise**2
        data = signal + variance**.5 * np.random.randn(self.shape[0], self.shape[1], self.shape[2])
        data[:,self.bad_map] = 0
        hot_ind = np.random.rand(self.shape[0], self.shape[1], self.shape[2]) < self.cr
        hot_amp = 1e4 + 1e6*np.random.rand(np.sum(hot_ind))
        data[hot_ind] = hot_amp
        
        return data, variance**.5


    def _psf(self, w, x):
        """Returns the PSF at column w and spatial offset x from trace
        centre. The PSF FWHM changes linearly with column coordinate
        (wavelength)
        """
        self.fwhm_scale = (self.end_fwhm-self.start_fwhm)/self.shape[2]
        s = (self.start_fwhm + w*self.fwhm_scale)/2.335
        return np.exp(-0.5*x**2/s**2)/(s*(2*np.pi)**.5)


    def _trace(self, w):
        """Define the spectral trace as row coordinates for
        input column coordinates. The trace is a quadratic
        function of column (wavelength)
        """
        v = w/self.shape[2]
        return 0.5*self.shape[1] + self.slope*(v-0.5) - v*(v-1)

        
    def _spectrum(self, w):
        """Produce a spectrum with random absorption lines
        and a baseline fall off with wavelength
        """
        v = w/self.shape[2]
        
        # Add absorption lines to spectrum
        Nlines = int(self.shape[2]/10)
        line_std = 0.5/Nlines
        tau = np.zeros(self.shape[2])
        max_tau = 0.3
        for n in range(Nlines):
            v0 = np.random.rand()
            tau0 = max_tau*np.random.rand()
            sigma = line_std*np.random.rand()
            tau += tau0*np.exp(-0.5*(v-v0)**2/sigma**2)
        
        # Let SED baseline fall off linearly
        baseline = 1 - (1-self.sed_fall)*v

        # Assume spectral resolution to be 2 pixels and convolve
        xx = np.linspace(-5,5,11)
        kern = np.exp(-xx**2)
        kern /= np.sum(kern)
            
        return np.convolve(baseline*np.exp(-tau), kern, mode='same')
            
        
