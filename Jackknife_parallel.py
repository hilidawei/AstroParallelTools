#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@time: 2024/10/02 17:14:23
@author: Dawei Li
@contact: llldawei@stu.xmu.edu.cn
"""

import numpy as np
import pandas
from astropy.io import fits
from astropy.table import Table
import math
from astropy.convolution import Gaussian2DKernel 
from astropy.convolution import convolve
import csv, os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from astropy.nddata.utils import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.coordinates import Angle, Latitude, Longitude
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as cosmo
import astropy.units as uu
from scipy import ndimage
import sys

from functools import partial
import multiprocessing
from multiprocessing import Pool, Manager, Process
import time

def scale_image(output_coords, scale, imwidth):
    mid = imwidth / 2
    return (output_coords[0] / scale + mid - mid / scale, output_coords[1] / scale + mid - mid / scale)

def jackknife_stacking(index, RA, Dec, Rf, imwidth):
    pd = []
    imwidth = imwidth
    for m in range(len(Rf)):
        dist = cosmo.kpc_proper_per_arcmin(Rf[m]).value
        pd.append(dist)
    pd = np.array(pd)
    Npx = int((np.min(pd)/np.max(pd)) * imwidth)
    print(Npx)
    # Select 90% of the sources randomly for each iteration
    sample_indices = np.random.choice(len(RA), int(0.9 * len(RA)), replace=False)

    stack_cts = 0
    stack_exp = 0

    for idx in sample_indices:
        # Load image and exposure map
        f1 = fits.open('../eROSITA_files/eFEDS_0.2-2.3_filtered_img.fits')
        f2 = fits.open('../eROSITA_files/eFEDS_0.2-2.3_filtered_exp.fits')
        cts = f1[0].data
        exp = f2[0].data
        wcs = WCS(f1[0].header)

        # Create cutouts around the source
        R0_cut = imwidth / 2
        cutout_cts = Cutout2D(cts, SkyCoord(RA[idx] * u.degree, Dec[idx] * u.degree, frame='fk5'), 2 * R0_cut, wcs=wcs)
        cutout_exp = Cutout2D(exp, SkyCoord(RA[idx] * u.degree, Dec[idx] * u.degree, frame='fk5'), 2 * R0_cut, wcs=wcs)

        # Rescale the images to a common frame based on redshift
        ps = cosmo.kpc_proper_per_arcmin(Rf[idx]).value
        scale = ps / np.max(pd)

        cts_img_rescaled = ndimage.geometric_transform(
            cutout_cts.data, scale_image, cval=0, order=0, extra_keywords={'scale': scale, 'imwidth': imwidth}
        )
        exp_img_rescaled = ndimage.geometric_transform(
            cutout_exp.data, scale_image, cval=0, order=0, extra_keywords={'scale': scale, 'imwidth': imwidth}
        )

        cts_img_cut = Cutout2D(cts_img_rescaled, (imwidth/2-0.5, imwidth/2-0.5), Npx)
        exp_img_cut = Cutout2D(exp_img_rescaled, (imwidth/2-0.5, imwidth/2-0.5), Npx)

        # Accumulate counts and exposure maps
        stack_cts += cts_img_cut.data[::]
        stack_exp += exp_img_cut.data[::]
    
    hdu = fits.PrimaryHDU(stack_cts)
    hdu.writeto(f'../test/stack_cts{index+1}.fits',overwrite=True)


    R_in = 1
    R_out = int(len(stack_cts)/2)

    R_bins = []
    for a in range(20):
        R_bins.append(int(pow(1.3,a)))
    R_bins = list(set(R_bins))
    R_bins.sort()
    bins = []
    for item in R_bins:
        if item < R_out:
            bins.append(item)
    bins.append(R_out)
    num_bins = len(bins) - 1

    # Initialize list to store surface brightness values from each Jackknife iteration
    jackknife_brightness_values = []

    # Calculate surface brightness profile for the stacked image
    xc, yc = stack_cts.shape[0] // 2, stack_cts.shape[1] // 2
    for j in range(num_bins):
        r_in = R_bins[j]
        r_out = R_bins[j + 1]

        y, x = np.ogrid[:stack_cts.shape[0], :stack_cts.shape[1]]
        r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        # Create a mask for the annulus
        annulus_mask = (r >= r_in) & (r < r_out)

        # Calculate the surface brightness in the annulus
        total_counts = np.sum(stack_cts[annulus_mask])
        total_exposure = np.sum(stack_exp[annulus_mask])
        total_bkg = 2.7244e-05 * total_exposure
        Rf_scale = [0.2, 0.3, 0.4, 0.5]
        az = (cosmo.kpc_proper_per_arcmin(0.2)/60).value 
        pixelsize = 4
        if total_exposure > 0:
            surface_brightness = (total_counts-total_bkg)/total_exposure/(pixelsize**2)/(az**2)*1e8
        else:
            surface_brightness = 0

        # Store the surface brightness value
        jackknife_brightness_values.append(surface_brightness)

    print(f"Jackknife iteration {index + 1} complete")

    return jackknife_brightness_values

if __name__ == "__main__":
    # Load catalog data
    zindex = 0
    mindex = 2
    dir_rs = ['0d1_0d2', '0d2_0d3', '0d3_0d4', '0d4_0d5']
    dir_mass = ['11d5_12', '12_12d5', '12d5_13', '13_13d5']
    imwidth = [300, 200, 140, 100]
    catalog = Table.read('../group_cata/'+dir_rs[zindex]+'/N234_'+dir_rs[zindex]+'_'+dir_mass[mindex]+'.csv')
    RA = np.array(catalog['RA'].data)
    Dec = np.array(catalog['Dec'].data)
    Rf = np.array(catalog['Redshift'].data)

    # Perform Jackknife stacking
    pool = Pool(6)
    num_jackknife = 50

    jackknife_partial = partial(jackknife_stacking, RA=RA, Dec=Dec, Rf=Rf, imwidth=imwidth[zindex])
    jackknife_results = pool.map(jackknife_partial, range(num_jackknife))

    # Save the jackknife results to CSV
    df = pandas.DataFrame(jackknife_results).transpose()
    df.columns = [f'Iteration_{i+1}' for i in range(num_jackknife)]
    df.to_csv('../Jackknife_profiles/Jackknife_profile_'+dir_rs[zindex]+'_'+dir_mass[mindex]+'.csv', index=False)
    print("Jackknife results saved to CSV")
