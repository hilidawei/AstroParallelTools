#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@time: 2024/10/13 15:55:35
@author: Dawei Li
@contact: llldawei@stu.xmu.edu.cn
"""

import time
import warnings
from astropy.utils.exceptions import AstropyWarning
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
from astropy.wcs.utils import skycoord_to_pixel
import time
from astropy.nddata.utils import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as cosmo
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

def scale_image(output_coords, scale, imwidth):
    mid = imwidth // 2
    return (output_coords[0] / scale + mid - mid / scale, output_coords[1] / scale + mid - mid / scale)


main_dir = '/Volumes/ldw_SSD/eRASS1_new/ero_archive/'
warnings.simplefilter('ignore', category=AstropyWarning)


image_paths = []
exp_paths = []
ra_dec_list = []


for base_name in os.listdir(main_dir):
    if not base_name.startswith('.') and os.path.isdir(os.path.join(main_dir, base_name)):
        base_dir = os.path.join(main_dir, base_name)

        for folder_name in os.listdir(base_dir):
            if not folder_name.startswith('.') and os.path.isdir(os.path.join(base_dir, folder_name)):
                folder_path = os.path.join(base_dir, folder_name)

                for subfolder_name in os.listdir(folder_path):
                    if not subfolder_name.startswith('.') and os.path.isdir(os.path.join(folder_path, subfolder_name)):
                        subfolder_path = os.path.join(folder_path, subfolder_name)

                        for file_name in os.listdir(subfolder_path):
                            if file_name.endswith('024_Image_c010.fits.gz') and not file_name.startswith('.'):
                                file_path = os.path.join(subfolder_path, file_name)
                                image_paths.append(file_path)
                                file_parts = file_name.split('_')
                                ra = file_parts[1][:3]
                                dec = file_parts[1][3:]
                                dec = 90 - float(dec)
                                ra_dec_list.append((ra, dec))
                            if file_name.endswith('024_ExposureMap_c010.fits.gz') and not file_name.startswith('.'):
                                file_path = os.path.join(subfolder_path, file_name)
                                exp_paths.append(file_path)


cata = Table.read('../data/0d5_0d6.csv')
imwidth = 300
RA = cata['RAdeg'].data
Dec = cata['DEdeg'].data
Rf = cata['zCl'].data


pd = [cosmo.kpc_proper_per_arcmin(Rf[i]) for i in range(len(Rf))]
Npx = int((min(pd) / max(pd)) * imwidth)


def process_source(i):
    global RA, Dec, pd, Npx, imwidth, ra_dec_list, image_paths, exp_paths

    if os.path.exists(f'../temp/cts_img_cut_{i}.npy') and os.path.exists(f'../temp/exp_img_cut_{i}.npy'):
        print(f"Source {i} already processed, skipping...")
        return True

    ra = RA[i]
    dec = Dec[i]
    for j in range(len(ra_dec_list)):
        if abs(ra - float(ra_dec_list[j][0])) < 7.2 and abs(dec - float(ra_dec_list[j][1])) < 7.2:
            original_image_filename = os.path.basename(image_paths[j])
            masked_image_filename = f'masked_{original_image_filename}'
            image_path = f'../eRASS1_masked_data/{masked_image_filename}'

            original_exp_filename = os.path.basename(exp_paths[j])
            modified_exp_filename = original_exp_filename.replace('ExposureMap', 'ExpMap')
            masked_exp_filename = f'masked_{modified_exp_filename}'
            exp_path = f'../eRASS1_masked_data/{masked_exp_filename}'


            try:
                with fits.open(image_path) as f1, fits.open(exp_path) as f2:
                    wcs = WCS(f1[0].header)
                    cts = f1[0].data
                    pixel_x, pixel_y = skycoord_to_pixel(SkyCoord(ra*u.deg, dec*u.deg, frame='fk5'), wcs)
                    

                    if 0 <= pixel_x < cts.shape[1] and 0 <= pixel_y < cts.shape[0]:
                        cutout = Cutout2D(cts, SkyCoord(ra*u.deg, dec*u.deg, frame='fk5'), imwidth, wcs, mode='partial', fill_value=0)
                        cts1 = cutout.data     

                        wcs2 = WCS(f2[0].header)
                        exp = f2[0].data
                        cutout2 = Cutout2D(exp, SkyCoord(ra*u.deg, dec*u.deg, frame='fk5'), imwidth, wcs2, mode='partial', fill_value=0)
                        exp1 = cutout2.data

                        ps = pd[i] / max(pd)
                        start_time = time.time()
                        cts_img = ndimage.geometric_transform(cts1, scale_image, cval=0, order=0, extra_keywords={'scale': ps, 'imwidth': imwidth})
                        exp_img = ndimage.geometric_transform(exp1, scale_image, cval=0, order=0, extra_keywords={'scale': ps, 'imwidth': imwidth})
                        end_time = time.time()

                        cts_img_cut = Cutout2D(cts_img, (imwidth/2 - 0.5, imwidth/2 - 0.5), Npx)
                        exp_img_cut = Cutout2D(exp_img, (imwidth/2 - 0.5, imwidth/2 - 0.5), Npx)
                        
                        np.save(f'../temp/cts_img_cut_{i}.npy', cts_img_cut.data)
                        np.save(f'../temp/exp_img_cut_{i}.npy', exp_img_cut.data)
                        print(f'Complete source {i}, time: {end_time - start_time:.2f}s')
                        

                        return True
                        break

            except Exception as e:
                print(f"Error processing source {i} (RA: {ra}, Dec: {dec}): {e}")
            
    return False

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=7) as executor:
        results = list(executor.map(process_source, range(len(RA))))

    print('Successfully processed all images')
